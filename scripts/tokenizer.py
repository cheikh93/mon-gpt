from collections import Counter

class SimpleTokenizer:
    def __init__(self, text, vocab_size=100):
        tokens = text.split()
        freq = Counter(tokens)
        self.vocab = {word: idx+1 for idx, (word, _) in enumerate(freq.most_common(vocab_size))}
        self.vocab['<PAD>'] = 0
        self.inv_vocab = {idx: word for word, idx in self.vocab.items()}

    def encode(self, text):
        return [self.vocab.get(word, 0) for word in text.split()]

    def decode(self, ids):
        return ' '.join([self.inv_vocab.get(i, '<UNK>') for i in ids])
