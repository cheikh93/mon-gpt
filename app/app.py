from flask import Flask, request, jsonify
import torch
from model.gpt import GPTMini
from scripts.tokenizer import SimpleTokenizer

# Charger le texte d'entraînement pour reconstruire le tokenizer
with open("data/corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Initialiser le tokenizer
tokenizer = SimpleTokenizer(text, vocab_size=100)

# Charger le modèle
model = GPTMini(
    vocab_size=len(tokenizer.vocab),
    embed_dim=64,
    n_heads=2,
    hidden_dim=128,
    n_layers=2,
    block_size=8
)
model.load_state_dict(torch.load("model/gpt_mini.pt"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Flask app
app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")

    # Encode
    input_ids = tokenizer.encode(prompt)[-8:]  # garder les 8 derniers mots max
    x = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        logits = model(x)
        next_token_id = logits[0, -1].argmax().item()

    next_word = tokenizer.decode([next_token_id])
    return jsonify({
        "prompt": prompt,
        "generated": next_word
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
