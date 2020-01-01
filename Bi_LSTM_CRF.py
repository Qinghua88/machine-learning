#### Use BiLSTM_CRF to do name entity recognition
#### BiLSTM generation features as the input of CRF(conditional random field)
#### CRF makes the transition matrix more reasonable than just using BiLSTM alone

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag2ind, embedding_d, hidden_d):
        super().__init__()
        self.embedding_d = embedding_d
        self.hidden_d = hidden_d
        self.tag2ind = tag2ind
        self.tags_d = len(tag2ind)

        self.word_embedding = nn.Embedding(vocab_size, embedding_d)
        self.lstm = nn.LSTM(embedding_d, hidden_d//2, num_layers=1, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_d, self.tags_d)
        # for transition matrix, transition[i][j] means the score of transition form j to i
        self.transition = nn.Parameter(torch.randn(self.tags_d, self.tags_d))
        self.transition.data[tag2ind[START_TAG], :] = -10000
        self.transition.data[:, tag2ind[END_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_d//2), torch.randn(2, 1, self.hidden_d//2))
        # (num_layers * num_directions, batch, hidden_size)


    def argmax(self, vec):
        _, index = torch.max(vec, 1)
        return index.item()

    def log_sum_exp(self, vec):
        max_score = vec[0, self.argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def _forward_algorithm(self, lstm_out):
        init_score = torch.full((1, self.tags_d), -10000)
        init_score[0][self.tag2ind[START_TAG]] = 0
        forward_score = init_score

        for out in lstm_out:
            score_t = []
            for next_tag in range(self.tags_d):
                emit_score = out[next_tag].view(1, -1).expand(1, self.tags_d)
                trans_score = self.transition[next_tag].view(1, -1)
                next_tag_score = forward_score + trans_score + emit_score
                score_t.append(self.log_sum_exp(next_tag_score).view(1))
            forward_score = torch.cat(score_t).view(1, -1)

        last_score = forward_score + self.transition[self.tag2ind[END_TAG]]
        z = self.log_sum_exp(last_score)
        return z

    def path_score(self, lstm_out, targets):
        score = torch.zeros(1) # score for the START_TAG
        tags = torch.cat([torch.tensor([self.tag2ind[START_TAG]], dtype=torch.long), targets])
        for i, out in enumerate(lstm_out):
            score = score + self.transition[tags[i + 1], tags[i]] + out[tags[i + 1]]
        score = score + self.transition[self.tag2ind[END_TAG], tags[-1]]
        return score

    def neg_log_likelihood(self, sentence, targets):
        lstm_out = self._get_lstm_features(sentence)
        z = self._forward_algorithm(lstm_out) # calculate partitioning score
        path_score = self.path_score(lstm_out, targets)
        return z - path_score




    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embedding = self.word_embedding(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embedding, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_d)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentence): # for model prediction
        lstm_out = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi(lstm_out) # find the best path, given the outcome from LSTM
        return score, tag_seq

    def _viterbi(self, lstm_out):
        back = []
        init_score = torch.full((1, self.tags_d), -10000)
        init_score[0][self.tag2ind[START_TAG]] = 0
        forward_score = init_score
        for out in lstm_out:
            back_t = []
            scores_t = []
            for next_tag in range(self.tags_d):
                next_tag_score = forward_score + self.transition[next_tag]
                best_tag_id = self.argmax(next_tag_score)
                back_t.append(best_tag_id)
                scores_t.append(next_tag_score[0][best_tag_id].view(1))
            forward_score = (torch.cat(scores_t) + out).view(1, -1)
            back.append(back_t)

        last_score = forward_score + self.transition[self.tag2ind[END_TAG]]
        best_tag_id = self.argmax(last_score)
        path_score = last_score[0][best_tag_id]

        best_path = [best_tag_id]
        for pointer in reversed(back):
            best_tag_id = pointer[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag2ind[START_TAG]
        best_path.reverse()
        return path_score, best_path



def prepare_sequence(sentence, word2ind):
    indexs = [word2ind[w] for w in sentence]
    return torch.tensor(indexs, dtype=torch.long)



START_TAG = '<s>'
END_TAG ='<E>'

if __name__ == "__main__":
    embedding_d = 5
    hidden_d = 4

    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]

    word2ind ={}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word2ind:
                word2ind[word] = len(word2ind)

    tag2ind = {'B': 0, 'I': 1, 'O': 2, START_TAG: 3, END_TAG: 4}
    model = BiLSTM_CRF(len(word2ind), tag2ind, embedding_d, hidden_d)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4) # what is weight_decay?



    for epoch in range(200):
        for sentence, tags in training_data:
            # since torch will accumulate the gradients all the way along, we clear it manually when needed
            model.zero_grad()

            sentence_in = prepare_sequence(sentence, word2ind)
            targets = torch.tensor([tag2ind[t] for t in tags], dtype=torch.long)
            # note that the targets contain no START_TAG or END_TAG

            loss = model.neg_log_likelihood(sentence_in, targets) # run forward pass

            loss.backward()
            optimizer.step() # update gradients and parameters

    with torch.no_grad():
        test_X = prepare_sequence(training_data[0][0], word2ind)
        print(test_X)
        print(model(test_X))

