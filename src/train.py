import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers  import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from tqdm import tqdm

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    # config['tokenizer_file']= f'.../tokenizers/tokenizer_{0}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]","[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer 

def get_ds(config):

    # Load dataset
    # ds_raw = load_dataset('iwslt2017', f"iwslt2017-{config['lang_src']}-{config['lang_tgt']}", split= 'train')
    ds_raw = load_dataset("opus_books", f"{config['lang_src']}-{config['lang_tgt']}", split="train")
    # ds_raw = load_dataset('opus100', f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    print("Dataset size:", len(ds_raw))
    
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Train/val split
    split = ds_raw.train_test_split(test_size=0.1, seed=42)
    train_raw = split["train"]
    val_raw = split["test"]

    print("Train:", len(train_raw))
    print("Val:", len(val_raw))

    # Build dataset objects
    train_ds = BilingualDataset(
        train_raw, tokenizer_src, tokenizer_tgt, 
        config['lang_src'], config['lang_tgt'], 
        config['seq_len']
    )

    valid_ds = BilingualDataset(
        val_raw, tokenizer_src, tokenizer_tgt, 
        config['lang_src'], config['lang_tgt'], 
        config['seq_len']
    )

    # Compute max lengths using the correct tokenizer
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_text = item['translation'][config['lang_src']]
        tgt_text = item['translation'][config['lang_tgt']]

        src_ids = tokenizer_src.encode(src_text).ids
        tgt_ids = tokenizer_tgt.encode(tgt_text).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length src: {max_len_src}")
    print(f"Max length tgt: {max_len_tgt}")

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(valid_ds,  batch_size=1)

    return train_loader, val_loader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'],config['d_model'])
    return model

def greedy_decoder(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Encode once
    encoder_output = model.encode(source, source_mask)

    # Start with SOS
    decoder_input = torch.tensor([[sos_idx]], dtype=torch.long, device=device)

    while True:
        # ---- Correct stop condition (use seq length) ----
        if decoder_input.size(1) >= max_len:
            break

        # Mask for current length
        decoder_mask = causal_mask(decoder_input.size(1)).to(device)

        # Decode
        output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Project and take last token logits
        prob = model.project(output)[:, -1, :]   # (1, vocab_size)
        next_word = torch.argmax(prob, dim=-1).item()

        # Append next token
        decoder_input = torch.cat(
            [decoder_input, torch.tensor([[next_word]], device=device)], dim=1
        )

        # Stop on EOS
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def validation_step(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples = 2):
    model.eval()
    count = 0

    # source_text_list = []
    # expected_list = []
    # prediction_list = []

    console_widt = 80

    with torch.no_grad():
        for batch in validation_ds:
            count +=1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "batch size must be 1 for validation"

            model_output = greedy_decoder(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]

            model_output_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            # source_text_list.append(source_text)
            # expected_list.append(target_text)
            # prediction_list.append(model_output_text)

            print_msg('-'*console_widt)
            print_msg(f'Source: {source_text}')
            print_msg(f'Target: {target_text}')
            print_msg(f'Predicted: {model_output_text}')

            if count == num_examples:
                break


def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(
        config, 
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size()
    ).to(device)

    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    # -----------------------------
    # LOAD CHECKPOINT IF AVAILABLE
    # -----------------------------
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')

        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1

        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    # -----------------------------
    # LOSS FUNCTION
    # -----------------------------
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id('[PAD]'),
        label_smoothing=0.1
    ).to(device)

    # -----------------------------
    #           TRAINING LOOP
    # -----------------------------
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Training epoch {epoch:02d}')

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            # forward
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                label.view(-1)
            )

            batch_iterator.set_postfix(loss=f"{loss.item():.4f}")
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # run validation at end of epoch
        validation_step(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # -----------------------------
        # SAVE CHECKPOINT EACH EPOCH
        # -----------------------------
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)



if __name__ == '__main__':
    # warnings.filterwarnings('ignore)
    config = get_config()
    train_model(config)

























