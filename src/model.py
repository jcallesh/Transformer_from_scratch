import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    """
    Embedding layer with scaling for transformer models.

    Wraps `nn.Embedding` and scales outputs by sqrt(d_model) (Sec. 3.4 in Vaswani et al., 2017).

    Args:
        d_model (int): Dimensionality of embeddings.
        vocabulary_size (int): Vocabulary size.

    Methods:
        forward(x):
            Maps token indices to scaled embeddings.

            Args:
                x (torch.Tensor): Input of shape (batch_size, sequence_length).

            Returns:
                torch.Tensor: Output of shape (batch_size, sequence_length, d_model).
    """
    def __init__(self, d_model, vocabulary_size):
        super().__init__()
        self.d_model    = d_model
        self.vocab_size = vocabulary_size
        self.embedding  = nn.Embedding(self.vocab_size, self.d_model)
        self.scale      = math.sqrt(self.d_model) 

    def forward(self, x):
        return self.embedding(x) * self.scale

class PositionalEconding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models (Sec. 3.5 in Vaswani et al., 2017).

    Adds fixed sine and cosine encodings to token embeddings to provide
    sequence order information. A dropout layer is applied for regularization.

    Args:
        d_model (int): Dimensionality of embeddings.
        seq_len (int): Maximum sequence length for which encodings are precomputed.
        dropout (float): Dropout probability applied after adding encodings.

    Methods:
        forward(x):
            Adds positional encodings to input embeddings and applies dropout.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, sequence_length, d_model).
    """
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pos_encoding = torch.zeros(self.seq_len, self.d_model)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() *
                             (-math.log(10000.0) / self.d_model))

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)  # shape: (1, seq_len, d_model)

        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        x = x + self.pos_encoding[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    """Applies layer normalization over the last dimension of the input.

    Normalizes inputs by subtracting the mean and dividing by the standard
    deviation, then applies learnable scaling (`alpha`) and bias parameters.

    Args:
        eps (float, optional): Small constant added to the denominator for
            numerical stability. Default is 1e-6.

    Methods:
        forward(x):
            Applies layer normalization.

            Args:
                x (torch.Tensor): Input tensor of shape
                    (batch_size, ..., feature_dim).

            Returns:
                torch.Tensor: Normalized tensor of the same shape as input.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps   = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias  = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mu    = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mu) / (sigma + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """Two-layer feedforward block with ReLU and dropout.

    Args:
        d_model (int): Input and output embedding dimension.
        d_ff (int): Hidden layer dimension.
        dropout (float): Dropout probability.

    Methods:
        forward(x):
            Applies linear → ReLU → dropout → linear.

            Args:
                x (torch.Tensor): Input of shape (batch_size, seq_len, d_model).

            Returns:
                torch.Tensor: Output of shape (batch_size, seq_len, d_model).
    """
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout  = nn.Dropout(dropout)   # fixed typo
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = torch.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
    
class MultiHeadAttentionBlock(nn.Module):
    """Multi-head attention block for transformer models.

    Splits queries, keys, and values into multiple heads, applies scaled
    dot-product attention, and projects the concatenated results.

    Args:
        d_model (int): Embedding dimension.
        h (int): Number of attention heads (must divide d_model).
        dropout (float): Dropout probability applied to attention weights.

    Methods:
        forward(q, k, v, mask):
            Applies multi-head attention.

            Args:
                q (torch.Tensor): Query input of shape (batch_size, seq_len, d_model).
                k (torch.Tensor): Key input of shape (batch_size, seq_len, d_model).
                v (torch.Tensor): Value input of shape (batch_size, seq_len, d_model).
                mask (torch.Tensor or None): Optional mask tensor.

            Returns:
                torch.Tensor: Output of shape (batch_size, seq_len, d_model).
    """
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h       = h
        self.dropout = nn.Dropout(dropout)   # fixed typo

        assert self.d_model % self.h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.W_q = nn.Linear(self.d_model, self.d_model) # Wk
        self.W_k = nn.Linear(self.d_model, self.d_model) # Wq
        self.W_v = nn.Linear(self.d_model, self.d_model) # Wv

        self.W_o = nn.Linear(self.d_model, self.d_model) # Wo
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores 


    def forward(self, q, k, v, mask):
         
        query = self.W_q(q) # this is Q \times W^Q
        key   = self.W_k(k)
        value = self.W_v(v)

        #Now we split each matrix int osmaller parts:
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key   = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x , self.attention_scores  = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.d_k)

        return self.W_o(x)
    
class ResidualConnection(nn.Module):
    """Residual connection with layer normalization and dropout.

    Applies layer normalization before a sublayer, then adds the sublayer
    output back to the original input with dropout for regularization.

    Args:
        dropout (float): Dropout probability applied to the sublayer output.

    Methods:
        forward(x, sublayer):
            Applies residual connection.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
                sublayer (Callable): Function or module applied to normalized input.

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
    """
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm    = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """Transformer encoder block with self-attention and feedforward layers.

    Combines a multi-head self-attention block and a feedforward block,
    each wrapped with residual connections and layer normalization.

    Args:
        self_attention_block (nn.Module): Multi-head self-attention module.
        feed_forward_block (nn.Module): Feedforward module.
        dropout (float): Dropout probability for residual connections.

    Methods:
        forward(x, src_mask):
            Applies self-attention and feedforward sublayers with residuals.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
                src_mask (torch.Tensor or None): Optional source mask.

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
    """
    def __init__(self, self_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block   = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """Transformer encoder composed of stacked encoder blocks.

    Applies a sequence of encoder layers to the input, followed by layer normalization.

    Args:
        layers (Callable): Function that returns a list or generator of encoder blocks.

    Methods:
        forward(x, mask):
            Applies all encoder blocks and final normalization.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
                mask (torch.Tensor or None): Optional source mask.

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
    """
    def __init__(self, layers):
        super().__init__()
        self.layers = layers      # ModuleList
        self.norm   = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    """Transformer decoder block with self-attention, cross-attention, and feedforward layers.

    Applies masked self-attention, encoder-decoder cross-attention, and a feedforward
    network, each wrapped in a residual connection with layer normalization and dropout.

    Args:
        self_attention_block (nn.Module): Masked multi-head self-attention module.
        cross_attention_block (nn.Module): Cross-attention module over encoder output.
        feed_forward_block (nn.Module): Feedforward module.
        dropout (float): Dropout probability for residual connections.

    Methods:
        forward(x, encoder_output, src_mask, tgt_mask):
            Applies decoder block operations.

            Args:
                x (torch.Tensor): Decoder input of shape (batch_size, tgt_seq_len, d_model).
                encoder_output (torch.Tensor): Encoder output of shape (batch_size, src_seq_len, d_model).
                src_mask (torch.Tensor or None): Optional source mask.
                tgt_mask (torch.Tensor or None): Optional target mask.

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, tgt_seq_len, d_model).
    """
    def __init__(self, self_attention_block, cross_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block  = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block    = feed_forward_block
        self.residual_connections  = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    """Transformer decoder composed of stacked decoder blocks.

    Applies a sequence of decoder layers with masked self-attention,
    cross-attention, and feedforward sublayers, followed by layer normalization.

    Args:
        layers (Callable): Function that returns a list or generator of decoder blocks.

    Methods:
        forward(x, encoder_output, src_mask, tgt_mask):
            Applies all decoder blocks and final normalization.

            Args:
                x (torch.Tensor): Decoder input of shape (batch_size, tgt_seq_len, d_model).
                encoder_output (torch.Tensor): Encoder output of shape (batch_size, src_seq_len, d_model).
                src_mask (torch.Tensor or None): Optional source mask.
                tgt_mask (torch.Tensor or None): Optional target mask.

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, tgt_seq_len, d_model).
    """
    def __init__(self, layers):
        super().__init__() 
        self.layers = layers # ModuleList
        self.norm   = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    """Final output layer projecting model embeddings to vocabulary logits.

    Applies a linear transformation followed by log-softmax over the vocabulary dimension.

    Args:
        d_model (int): Dimensionality of model embeddings.
        vocab_size (int): Size of the output vocabulary.

    Methods:
        forward(x):
            Projects embeddings to log-probabilities over the vocabulary.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

            Returns:
                torch.Tensor: Log-probabilities of shape (batch_size, seq_len, vocab_size).
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.proj(x)
        return torch.log_softmax(x, dim = -1)



class Transformer(nn.Module):
    """Full transformer model combining encoder, decoder, embeddings, and output projection.

    Encodes a source sequence and decodes a target sequence using stacked attention blocks,
    positional encodings, and a final projection to vocabulary logits.

    Args:
        encoder (nn.Module): Transformer encoder module.
        decoder (nn.Module): Transformer decoder module.
        src_embedding (nn.Module): Token embedding layer for the source input.
        tgt_embedding (nn.Module): Token embedding layer for the target input.
        src_position (nn.Module): Positional encoding for the source input.
        tgt_position (nn.Module): Positional encoding for the target input.
        projection_layer (nn.Module): Final linear layer projecting to vocabulary size.

    Methods:
        encode(src, src_mask):
            Embeds and encodes the source sequence.

            Args:
                src (torch.Tensor): Source input of shape (batch_size, src_seq_len).
                src_mask (torch.Tensor or None): Optional source mask.

            Returns:
                torch.Tensor: Encoded source of shape (batch_size, src_seq_len, d_model).

        decode(encoder_output, src_mask, tgt, tgt_mask):
            Embeds and decodes the target sequence using encoder output.

            Args:
                encoder_output (torch.Tensor): Output from the encoder.
                src_mask (torch.Tensor or None): Optional source mask.
                tgt (torch.Tensor): Target input of shape (batch_size, tgt_seq_len).
                tgt_mask (torch.Tensor or None): Optional target mask.

            Returns:
                torch.Tensor: Decoder output of shape (batch_size, tgt_seq_len, d_model).

        project(x):
            Projects decoder output to vocabulary logits.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

            Returns:
                torch.Tensor: Logits of shape (batch_size, seq_len, vocab_size).
    """
    def __init__(self, encoder, decoder, src_embedding, tgt_embedding, src_position, tgt_position, projection_layer):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_position  = src_position
        self.tgt_position  = tgt_position
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_position(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_position(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask) 

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model = 512, Nx = 6, h = 8, dropout = 0.1, d_ff = 2048 ):
    """Builds a full transformer model with encoder, decoder, embeddings, and output projection.

    Constructs a transformer architecture using stacked attention blocks, positional encodings,
    and feedforward layers. Initializes parameters with Xavier uniform distribution.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        src_seq_len (int): Maximum source sequence length.
        tgt_seq_len (int): Maximum target sequence length.
        d_model (int, optional): Embedding dimension. Defaults to 512.
        Nx (int, optional): Number of encoder and decoder blocks. Defaults to 6.
        h (int, optional): Number of attention heads. Defaults to 8.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        d_ff (int, optional): Hidden dimension of feedforward layers. Defaults to 2048.

    Returns:
        Transformer: A fully constructed transformer model.
    """

    # create embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size) 
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # create the positional encoding layers
    src_pos = PositionalEconding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEconding(d_model, tgt_seq_len, dropout) #this doesn't have learning parameters so we could use src_pos instead!

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(Nx):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks  
    decoder_blocks = []
    for _ in range(Nx):
        decoder_self_attention_block  = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # creathe the output layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters 
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer












