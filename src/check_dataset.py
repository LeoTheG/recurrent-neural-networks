from collections import Counter
from torchtext.datasets import PennTreebank

# Load the Penn Treebank dataset
train_data, valid_data, test_data = PennTreebank()

train_sentences = [sentence.split() for sentence in train_data]
valid_sentences = [sentence.split() for sentence in valid_data]
test_sentences = [sentence.split() for sentence in test_data]

# Build a vocabulary
counter = Counter(word for sentence in train_sentences for word in sentence)

print("First sentence in train_data:")
for i, item in enumerate(train_data):
    if i == 0:
        print(item)
    else: break

print("Most common tokens:")
for token, count in counter.most_common(10):  # top 10 tokens
    print(token, count)