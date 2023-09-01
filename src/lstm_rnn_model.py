import torch.nn as nn

models_directory = 'models/'

class LSTMRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # Using batch_first for convenience
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embeds = self.embedding(x)
        # embeds shape: (batch_size, sequence_length, embedding_dim)
        lstm_out, _ = self.lstm(embeds)
        # lstm_out shape: (batch_size, sequence_length, hidden_dim)
        output_space = self.linear(lstm_out.contiguous().view(-1, self.hidden_dim))
        # output_space shape: (batch_size*sequence_length, output_dim)
        return output_space