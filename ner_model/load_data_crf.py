import pandas as pd


data = pd.read_csv("./ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")

words = list(set(data["Word"].values))
words.append("UNKNOWN")
words.append("ENDPAD")
words.sort()
words.remove('None')
num_words = len(words)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
tags = list(set(data["Tag"].values))
tags.append(START_TAG)
tags.append(STOP_TAG)
tags.sort()
num_tags = len(tags)

agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(), s["Tag"].values.tolist())]
grouped = data.groupby("Sentence #").apply(agg_func)
sentences = [s for s in grouped]

word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

X = []
for s in sentences:
    cur = []
    for w in s:
        if w[0] in word2idx:
            cur.append(word2idx[w[0]])
        else:
            cur.append(word2idx['Unknown'])
    X.append(cur)
y = [[tag2idx[w[1]] for w in s] for s in sentences]

test_proportion = 0.2
train_size = int((1-0.2) * len(X))
train_X, train_y, test_X, test_y = X[0:train_size], y[0:train_size], X[train_size:], y[train_size:]