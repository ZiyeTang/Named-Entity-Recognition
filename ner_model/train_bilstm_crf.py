import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import random
from bilstm_crf import START_TAG, STOP_TAG
from bilstm_crf import BiLSTM_CRF
from load_data_crf import num_words, num_tags, words, tags, word2idx, tag2idx, train_X, train_y, test_X, test_y, train_size

device = (
    "cuda"
    if torch.backends.mps.is_available()
    else "cpu"
)

def train(data_X, data_Y, model, loss_fn, optimizer, batch_size):
    model.train()
    temp = list(zip(data_X, data_Y))
    random.shuffle(temp)
    data_X, data_Y = zip(*temp)
    data_X, data_Y = list(data_X), list(data_Y)
    for i in range(0, len(data_X), batch_size):
        cur_X, cur_y = data_X[i:min(i+batch_size, train_size)], data_Y[i:min(i+batch_size, train_size)]
        cur_X = pad_sequence([torch.tensor(s) for s in cur_X],  batch_first=True, padding_value=word2idx['ENDPAD']).to(device)
        cur_y = pad_sequence([torch.tensor(s) for s in cur_y],  batch_first=True, padding_value=tag2idx['O']).to(device)
        # Compute prediction error
        # pred = model(cur_X)
        loss = model.neg_log_likelihood(cur_X, cur_y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i//batch_size+1) % 1000 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}")

model = BiLSTM_CRF(num_words, tag2idx, embedding_dim=50, hidden_dim=200).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



epochs = 10
for t in range(epochs):
    print(f"\nEpoch {t+1}\n-------------------------------")
    train(train_X, train_y, model, loss_fn, optimizer, 5)

torch.save(model.state_dict(), "bilstm_crf.pth")