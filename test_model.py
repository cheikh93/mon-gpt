from model.gpt import GPTMini
import torch

vocab_size = 50
model = GPTMini(vocab_size, embed_dim=32, n_heads=2, hidden_dim=64, n_layers=2, block_size=8)

x = torch.randint(0, vocab_size, (1, 8))  # batch de 1, séquence de 8
out = model(x)
print("Shape sortie :", out.shape)  # Doit afficher torch.Size([1, 8, 50])
