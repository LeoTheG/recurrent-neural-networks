'''
Notes: 
To apply chunking based on the sequence length, we need to modify the way sentences are fed into the model for training. The logic would be:

1. Flatten the list of sentences into a single sequence.
2. Split this sequence into chunks of the given SEQUENCE_LENGTH.
'''
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torchtext.datasets import PennTreebank
import json
from lstm_rnn_model import models_directory, LSTMRNN

# Load the Penn Treebank dataset
train_data, valid_data, test_data = PennTreebank()

train_sentences = [sentence.split() for sentence in train_data]
valid_sentences = [sentence.split() for sentence in valid_data]
test_sentences = [sentence.split() for sentence in test_data]

# Build a vocabulary
counter = Counter(word for sentence in train_sentences for word in sentence)

vocab = sorted(counter, key=counter.get, reverse=True)
VOCAB_SIZE = len(vocab)
token_to_idx = {token: idx for idx, token in enumerate(vocab)}
idx_to_token = {idx: token for token, idx in token_to_idx.items()}

# save the token_to_idx and idx_to_token mappings for later use
with open('token_to_idx.json', 'w') as f:
    json.dump(token_to_idx, f)

with open('idx_to_token.json', 'w') as f:
    json.dump(idx_to_token, f)

# Some constants
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
EPOCHS = 200
EPOCH_SAVE_INTERVAL = 25
SEQUENCE_LENGTH = 30

def prepare_sequence(seq, to_idx):
    return torch.tensor([to_idx[s] for s in seq], dtype=torch.long)

# Instantiate the model, loss, and optimizer
model = LSTMRNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("GPU not detected. Exiting...")
    exit()
model.to(device)

# Flatten the train_sentences for chunking
flattened_data = [word for sentence in train_sentences for word in sentence]

# Train the model using chunks
for epoch in range(EPOCHS):
    total_loss = 0
    for i in range(0, len(flattened_data) - SEQUENCE_LENGTH, SEQUENCE_LENGTH): 
        model.zero_grad()
        
        chunk = flattened_data[i:i+SEQUENCE_LENGTH]
        
        sentence_in = prepare_sequence(chunk[:-1], token_to_idx).to(device)
        targets = prepare_sequence(chunk[1:], token_to_idx).to(device)

        log_probs = model(sentence_in)
        
        loss = loss_function(log_probs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss}")

    if (epoch + 1) % EPOCH_SAVE_INTERVAL == 0:
        torch.save(model.state_dict(), f"{models_directory}lstm_epoch_{epoch+1}.pth")

torch.save(model.state_dict(), f'{models_directory}lstm.pth')