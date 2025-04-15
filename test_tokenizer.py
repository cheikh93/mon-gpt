from scripts.tokenizer import SimpleTokenizer

with open("data/corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = SimpleTokenizer(text)
print("Texte original :", text)
encoded = tokenizer.encode("Le chat dort")
print("Encodé :", encoded)
print("Décodé :", tokenizer.decode(encoded))
