import torch
import json
from lstm_rnn_model import LSTMRNN, models_directory

# Assuming your vocabulary and other constants are the same as in the training script
# We re-declare these constants for the sake of completion in this prediction file
EMBEDDING_DIM = 128
HIDDEN_DIM = 256

with open('token_to_idx.json', 'r') as f:
    token_to_idx = json.load(f)

with open('idx_to_token.json', 'r') as f:
    idx_to_token = {int(k): v for k, v in json.load(f).items()}

VOCAB_SIZE = len(token_to_idx)

def prepare_sequence(seq, to_idx):
    return torch.tensor([to_idx.get(s, 0) for s in seq], dtype=torch.long)  # using 0 for unknown tokens

def predict_next_token(model, token_sequence):
    with torch.no_grad():
        inputs = prepare_sequence(token_sequence, token_to_idx)
        log_probs = model(inputs)
        return idx_to_token[log_probs[-1].argmax(dim=0).item()]

def generate_sequence(model, start_tokens, num_tokens=50):
    generated_tokens = list(start_tokens)
    for i in range(num_tokens):
        next_token = predict_next_token(model, generated_tokens)
        generated_tokens.append(next_token)
    return generated_tokens

if __name__ == "__main__":
    # Load the model
    model = LSTMRNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE)
    model.load_state_dict(torch.load(f"{models_directory}lstm_epoch_90.pth"))
    model.eval()

    # Make a prediction
    # starting_tokens = ["The", "quick"]
    starting_tokens = ["One", "of"]
    generated = generate_sequence(model, starting_tokens, num_tokens=50)
    print(" ".join(generated))
