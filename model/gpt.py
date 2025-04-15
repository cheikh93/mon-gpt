import torch
import torch.nn as nn

class GPTMini(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, hidden_dim, n_layers, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)

        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, n_heads, hidden_dim, batch_first=True)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.head(x)
