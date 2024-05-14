import pandas as pd
data = pd.read_csv("./ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")

words = list(set(data["Word"].values))
words.sort()
words.append("UNKNOWN")
words.append("ENDPAD")
num_words = len(words)

tags = list(set(data["Tag"].values))
tags.sort()
num_tags = len(tags)

agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(), s["Tag"].values.tolist())]
grouped = data.groupby("Sentence #").apply(agg_func)
sentences = [s for s in grouped]

word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

X = [[word2idx[w[0]] for w in s] for s in sentences]
y = [[tag2idx[w[1]] for w in s] for s in sentences]

test_proportion = 0.2
train_size = int((1-0.2) * len(X))
train_X, train_y, test_X, test_y = X[0:train_size], y[0:train_size], X[train_size:], y[train_size:]