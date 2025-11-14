import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    """Custom dataset for bilingual translation tasks using tokenized parallel corpora.

    Prepares source and target sequences with special tokens (SOS, EOS, PAD),
    applies tokenization, and returns inputs, masks, and labels for training.

    Args:
        ds (Dataset): HuggingFace-style dataset with 'translation' field.
        tokenizer_src (Tokenizer): Tokenizer for the source language.
        tokenizer_tgt (Tokenizer): Tokenizer for the target language.
        src_lang (str): Source language code (e.g., 'en').
        tgt_lang (str): Target language code (e.g., 'es').
        seq_len (int): Fixed sequence length for padding and truncation.

    Methods:
        __len__():
            Returns the number of samples in the dataset.

        __getitem__(index):
            Returns tokenized and padded tensors for a given sample.

            Args:
                index (int): Index of the sample.

            Returns:
                dict: Dictionary with keys:
                    - encoder_input (torch.Tensor): Source input sequence.
                    - decoder_input (torch.Tensor): Target input sequence.
                    - encoder_mask (torch.Tensor): Padding mask for encoder.
                    - decoder_mask (torch.Tensor): Padding + causal mask for decoder.
                    - label (torch.Tensor): Target sequence with EOS for loss.
                    - src_text (str): Original source sentence.
                    - tgt_text (str): Original target sentence.
    """
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # Store IDs (ints, not tensors)
        self.sos_id_src = tokenizer_src.token_to_id("[SOS]")
        self.eos_id_src = tokenizer_src.token_to_id("[EOS]")
        self.pad_id_src = tokenizer_src.token_to_id("[PAD]")

        self.sos_id_tgt = tokenizer_tgt.token_to_id("[SOS]")
        self.eos_id_tgt = tokenizer_tgt.token_to_id("[EOS]")
        self.pad_id_tgt = tokenizer_tgt.token_to_id("[PAD]")

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):

        pair = self.ds[index]
        src_text = pair["translation"][self.src_lang]
        tgt_text = pair["translation"][self.tgt_lang]

        enc_tokens = self.tokenizer_src.encode(src_text).ids
        dec_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_pad = self.seq_len - len(enc_tokens) - 2
        dec_pad = self.seq_len - len(dec_tokens) - 1

        if enc_pad < 0 or dec_pad < 0:
            raise ValueError("Sentence too long")

        # Encoder input: [SOS] src [EOS] PAD*
        encoder_input = torch.cat([
            torch.tensor([self.sos_id_src], dtype=torch.int64),
            torch.tensor(enc_tokens, dtype=torch.int64),
            torch.tensor([self.eos_id_src], dtype=torch.int64),
            torch.full((enc_pad,), self.pad_id_src, dtype=torch.int64)
        ])

        # Decoder input: [SOS] tgt PAD*
        decoder_input = torch.cat([
            torch.tensor([self.sos_id_tgt], dtype=torch.int64),
            torch.tensor(dec_tokens, dtype=torch.int64),
            torch.full((dec_pad,), self.pad_id_tgt, dtype=torch.int64)
        ])

        # Label: tgt [EOS] PAD*
        label = torch.cat([
            torch.tensor(dec_tokens, dtype=torch.int64),
            torch.tensor([self.eos_id_tgt], dtype=torch.int64),
            torch.full((dec_pad,), self.pad_id_tgt, dtype=torch.int64)
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        encoder_mask = (encoder_input != self.pad_id_src).unsqueeze(0).unsqueeze(0)
        decoder_mask = (
            (decoder_input != self.pad_id_tgt).unsqueeze(0).unsqueeze(0)
            & causal_mask(self.seq_len)
        )

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(seq_len):
    mask = torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).type(torch.int)
    return mask == 0