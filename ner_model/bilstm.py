from torch import nn

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, tag2idx, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag2idx = tag2idx
        self.tagset_size = len(tag2idx)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim // 2,
                            num_layers = 1, batch_first = True, dropout = 0.1, bidirectional = True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

    def forward(self, x):
        output = self.word_embeds(x)
        output, _ = self.lstm(output)
        output =  self.hidden2tag(output)
        return output