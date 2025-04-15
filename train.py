import torch
from torch.utils.data import DataLoader
from scripts.tokenizer import SimpleTokenizer
from scripts.dataset import TextDataset
from model.gpt import GPTMini
from tqdm import tqdm

# Chargement des données
with open("data/corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Initialisation
tokenizer = SimpleTokenizer(text, vocab_size=100)
dataset = TextDataset(text, tokenizer, block_size=8)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTMini(
    vocab_size=len(tokenizer.vocab),
    embed_dim=64,
    n_heads=2,
    hidden_dim=128,
    n_layers=2,
    block_size=8
).to(device)

# Optimiseur + fonction de perte
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# Entraînement
EPOCHS = 10
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for x, y in progress:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    print(f"Époque {epoch+1} terminée | Loss moyenne : {total_loss / len(loader):.4f}")

# Sauvegarde du modèle
torch.save(model.state_dict(), "model/gpt_mini.pt")
