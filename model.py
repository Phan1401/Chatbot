import torch
import torch.nn as nn
import torch.nn.functional as F

from create_vocab import word_vocab

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_len, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=word_vocab["<pad>"])
        self.positional_encoding = self._generate_positional_encoding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.decoder = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _generate_positional_encoding(self, max_len, embed_dim):
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, embed_dim)

    def forward(self, x, src_mask=None):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        x = self.encoder(x, src_mask)
        return self.decoder(x)

    def generate_text(self, x, max_len=10):
        outputs = []
        for _ in range(max_len):
            x = self.forward(x).argmax(dim=-1)
            outputs.append(x)
            if (x == word_vocab["<pad>"]).all():
                break  # Dừng nếu toàn bộ câu là <pad>
        return torch.cat(outputs, dim=1)  # (batch_size, max_len)
