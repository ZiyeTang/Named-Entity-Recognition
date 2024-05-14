import torch
from torch import nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
START_TAG = "<START>"
STOP_TAG = "<STOP>"

def log_sum_exp(vec):
    res = 0
    for i in range(len(vec)):
        max_score = torch.max(vec[i])
        res+= max_score + torch.log(torch.sum(torch.exp(vec - max_score)))
    return res

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return [i.item() for i in idx]

class BiLSTM_CRF(nn.Module):
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
        
        
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag2idx[START_TAG], :] = -10000
        self.transitions.data[:, tag2idx[STOP_TAG]] = -10000

    def forward(self, x):
        feats = self._get_lstm_features(x)
        score, tag_seq = self._viterbi_decode(feats)
        return score, tag_seq
    
    def _get_lstm_features(self, sentences):
        output = self.word_embeds(sentences)
        output, _ = self.lstm(output)
        output =  self.hidden2tag(output)
        return output
    
    def _viterbi_decode(self, feats):
        backpointers = []

        init_vvars = torch.full((len(feats), self.tagset_size), -10000.)
        init_vvars[:, self.tag2idx[START_TAG]] = 0

        forward_vars = init_vvars
        for i in range(len(feats)):
            cur_bp = []
            for feat in feats[i]:
                bptrs_t = []  
                viterbivars_t = []  
                for next_tag in range(self.tagset_size):
                    next_tag_var = forward_vars[i:i+1] + self.transitions[next_tag]
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id[0]].view(1))
                forward_vars[i] = (torch.cat(viterbivars_t) + feat).view(1, -1)
                cur_bp.append(bptrs_t)
            backpointers.append(cur_bp)
        
        terminal_var = forward_vars + self.transitions[self.tag2idx[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = torch.tensor([terminal_var[i][best_tag_id[i]] for i in range(len(best_tag_id))])

        best_path = [[id] for id in best_tag_id]
        for i in range(len(backpointers)):
            for bptrs_t in reversed(backpointers[i]):
                best_tag_id[i] = bptrs_t[best_tag_id[i]][0]
                best_path[i].append(best_tag_id[i])
        
        start = [best_path[i].pop() for i in range(len(best_path))]
        assert start == [self.tag2idx[START_TAG]]*len(best_path)  # Sanity check
        
        for i in range(len(best_path)):
            best_path[i].reverse()

        return path_score, best_path


    def _score_sentence(self, feats, tgs):
        score = torch.zeros(1).to(device)
        for j in range(len(tgs)):
            temp = torch.cat([torch.tensor([self.tag2idx[START_TAG]], dtype=torch.long).to(device), tgs[j].to(device)]).to(device)
            for i, feat in enumerate(feats[j]):
                score += self.transitions[temp [i + 1], temp [i]].to(device) + feat[temp [i + 1]].to(device)
            score = score + self.transitions[self.tag2idx[STOP_TAG], temp [-1]].to(device)
        return score
    

    
    def _forward_alg(self, feats):
        init_alphas = torch.full((len(feats), self.tagset_size), -10000.)
        init_alphas[:, self.tag2idx[START_TAG]] = 0.

        forward_vars = init_alphas

        for i in range(len(feats)):
            for feat in feats[i]:
                alphas_t = [] 
                for next_tag in range(self.tagset_size):
                    emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                    trans_score = self.transitions[next_tag].view(1, -1)
                    next_tag_var = forward_vars[i].to(device) + trans_score.to(device) + emit_score.to(device)
                    alphas_t.append(log_sum_exp(next_tag_var).view(1))
                    
                forward_vars[i] = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_vars.to(device) + self.transitions[self.tag2idx[STOP_TAG]].to(device)
        alpha = log_sum_exp(terminal_var)
        return alpha

    def neg_log_likelihood(self, sentence, tgs):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tgs)
        return forward_score - gold_score
