from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

import torch
import pandas as pd
from ner_model.bilstm_crf import BiLSTM_CRF, START_TAG, STOP_TAG

data = pd.read_csv("./ner_model/ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")

words = list(set(data["Word"].values))
words.append("UNKNOWN")
words.append("ENDPAD")
words.sort()
words.remove('None')
num_words = len(words)


tags = list(set(data["Tag"].values))
tags.append(START_TAG)
tags.append(STOP_TAG)
tags.sort()
num_tags = len(tags)

word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

model = BiLSTM_CRF(num_words, tag2idx, embedding_dim=50, hidden_dim=200)
model.load_state_dict(torch.load("./ner_model/weights/bilstm_crf.pth", map_location='cpu'))

@app.route('/', methods=["POST"])
def inference():
    input = ""
    if len(list(request.form)) > 0:
        input= list(request.form)[0] 
    
    sentences = input.split(".")
    X = []
    words_list = []
    for s in sentences:
        cur = []
        cur_list = []
        splitedsent = s.split(' ')
        for w in splitedsent:
            if w == '':
                continue

            sidx = 0
            while sidx< len(w) and (w[sidx]<'A' or w[sidx]>'z' or (w[sidx]>'Z' and w[sidx] < 'a')):
                if w[sidx] in word2idx:
                    cur.append(word2idx[w[sidx]])
                else:
                    cur.append(word2idx['Unknown'])
                cur_list.append(w[sidx])
                sidx+=1
            
            suffix = []
            sufx_list = []
            eidx = len(w)-1   
            while eidx > sidx and (w[eidx]<'A' or w[eidx]>'z' or (w[eidx]>'Z' and w[eidx] < 'a')):
                if w[sidx] in word2idx:
                    suffix.append(word2idx[w[eidx]])
                else:
                    suffix.append(word2idx['Unknown'])
                sufx_list.append(w[eidx])
                eidx-=1

            if w[sidx: eidx+1] in word2idx:
                cur.append(word2idx[w[sidx: eidx+1]])
            else:
                cur.append(word2idx['Unknown'])
            cur_list.append(w[sidx: eidx+1])

            suffix.reverse()
            cur+=suffix
            sufx_list.reverse()
            cur_list+=sufx_list
        if len(cur) != 0:
            X.append(cur)
            words_list.append(cur_list)
    
    # print('\n',sentences,'\n')
    preds = []
    model.eval()
    for i in range(0, len(X)):
        cur_X = torch.tensor(X[i:i+1], dtype = torch.long)

        _, pred = model(cur_X)
        pred_tag = []
        for j in range(0, len(cur_X[0])):
            pred_tag.append(tags[pred[0][j]])
        
        preds.append(pred_tag)
    
    res = ""
    for i in range(len(X)):
        for j in range(len(preds[i])):
            res+=words_list[i][j]
            if preds[i][j]!='O' and (len(words_list[i][j])>1 or (words_list[i][j] >='A' and words_list[i][j] <= 'Z' and words_list[i][j] >= 'a' and words_list[i][j] <= 'z')):
                res+='('
                res+=preds[i][j]
                res+=')'
            if words_list[i][j]!='\n':
                res+=' '
    return res

if __name__=='__main__':
    app.run(debug=True)

