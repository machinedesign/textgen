from datetime import datetime

from clize import run
from molecules.transformers import DocumentVectorizer
import numpy as np
from collections import defaultdict
import pandas as pd
 
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

import spacy

from visdom import Visdom


UNK_CHAR = 3

def get_acc(pred, true):
    _, pred_classes = pred.max(1)
    acc = (pred_classes == true).float().mean()
    return acc


class RNN(nn.Module):

    def __init__(self, vocab_size=10, emb_size=100, hidden_size=128, latent_size=10, word_dropout=0, unk_char=UNK_CHAR, feed_latent=False):
        super(RNN, self).__init__()
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.word_dropout = word_dropout
        self.unk_char = unk_char
        self.feed_latent = feed_latent

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.enc_lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.latent = nn.Linear(hidden_size, latent_size * 2)
        self.init_hid = nn.Linear(latent_size, hidden_size)
        self.init_c = nn.Linear(latent_size, hidden_size)
        self.dec_lstm = nn.LSTM(emb_size + (latent_size if feed_latent else 0), hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, len_x):
        xe = self.emb(x)
        packed_x = rnn_utils.pack_padded_sequence(xe, len_x, batch_first=True)
        _, (h, c) = self.enc_lstm(packed_x)
        l = self.latent(h[-1]) # h[-1] = last hidden state of last lstm layer
        l_mean, l_log_var = l[:, 0:self.latent_size], l[:, self.latent_size:]
        l_std = torch.exp(l_log_var / 2.)
        l = l_mean + l_std * Variable(torch.randn(l_std.size())).cuda()
        init_h = self.init_hid(l)
        init_c = self.init_c(l)
        init_h = init_h.view(1, init_h.size(0), init_h.size(1))
        init_c = init_c.view(1, init_c.size(0), init_c.size(1))
        
        if self.word_dropout > 0:
            x_noise = word_dropout(x, self.word_dropout, unk_char=self.unk_char)
            x_noise = self.emb(x_noise)
            if self.feed_latent:
                l_ = l.view(l.size(0), 1, l.size(1)).repeat(1, xe.size(1), 1)
                xe_noise = torch.cat((x_noise, l_), 2)
            else:
                xe_noise = x_noise
            packed_x_noise = rnn_utils.pack_padded_sequence(xe_noise, len_x, batch_first=True)
        else:
            if self.feed_latent:
                l_ = l.view(l.size(0), 1, l.size(1)).repeat(1, xe.size(1), 1)
                xe_ = torch.cat((xe, l_), 2)
                packed_x_noise = rnn_utils.pack_padded_sequence(xe_, len_x, batch_first=True)
            else:
                packed_x_noise = packed_x
        packed_o, _ = self.dec_lstm(packed_x_noise, (init_h, init_c))
        o, len_o = rnn_utils.pad_packed_sequence(packed_o, batch_first=True)
        o = o.contiguous()
        o = self.out(o.view(-1, o.size(2)))
        return o, (l_mean, l_log_var, l)
    
    def greedy_generate(self, l, length=10, deterministic=True):
        l.register_hook(lambda g:print(torch.abs(g).mean().data[0]))
        init_h = self.init_hid(l)
        init_c = self.init_c(l)
        init_h = init_h.view(1, init_h.size(0), init_h.size(1))
        init_c = init_c.view(1, init_c.size(0), init_c.size(1))
        x = torch.ones(l.size(0), 1).long() # BEGIN CHARACTER  = 1
        x = Variable(x)
        x = x.cuda()
        out = torch.zeros(l.size(0), length).long()
        h, c = init_h, init_c
        l_ = l.view(l.size(0), 1, l.size(1))
        for i in range(length):
            xe = self.emb(x)
            if self.feed_latent:
               xe = torch.cat((xe, l_), 2)
            o, (h, c) = self.dec_lstm(xe, (h, c))
            #h, c = h.detach(), c.detach()
            o = self.out(o.view(o.size(0), -1))
            if deterministic:
                _, o = o.max(1)
            else:
                o = torch.multinomial(nn.Softmax()(o))
            x.data.copy_(o.data)
            o = o.data.cpu().long()
            out[:, i].copy_(o.view(-1))
        #torch.abs(h).mean().backward()
        return out


def word_dropout(x, pr, unk_char=0):
    mask = (torch.rand(x.size()) <= pr).long()
    mask = Variable(mask)
    mask = mask.cuda()
    x = x * (1 - mask) + mask * unk_char
    return x


def train(*, dataset='en.txt', length=10, hidden_size=200, batch_size=128, emb_size=350, lr=1e-3, nb_epochs=1000000, latent_size=20, word_dropout=0., feed_latent=False, folder='out'):
    np.random.seed(42)
    viz = Visdom('http://romeo163')
    win = viz.line(
        X=np.array([0]), 
        Y=np.array([0]), 
        opts=dict(title='textgen, started at {}, folder={}'.format(datetime.now(), folder)))
    viz.line(X=np.array([0]), Y=np.array([0]), update='append', win=win)
    print('reading corpus...')
    corpus = open(dataset).read()
    print('tokenizing...')
    nlp = spacy.load('en')
    doc = nlp(corpus)
    corpus = [sent for sent in doc.sents if len(sent) <= length]
    corpus = [[tok.string.strip() for tok in sent[0:-1]] for sent in corpus]
    corpus = corpus[0:100000]
    print('Size of corpus : {}'.format(len(corpus)))
    print(corpus[0:5])
    # the max length is +2 because we pad with the first and en character
    max_length = max(map(len, corpus)) + 2
    # Fitting document vectorizer
    print('Fitting document vectorizer...')
    doc = DocumentVectorizer(pad=True, begin_character=True, end_character=True, length=max_length)
    doc._update(set([UNK_CHAR]))
    doc.partial_fit(corpus)
    vocab_size = len(doc.words_)
    print('vocab size : {}'.format(vocab_size))
    # Model
    rnn = RNN(
        vocab_size=vocab_size,
        emb_size=emb_size, # embedding size
        hidden_size=hidden_size, # hidden size of encoder and decoder 
        latent_size=latent_size, # latent Z size
        word_dropout=word_dropout,
        feed_latent=feed_latent) # whether to feed latent in each decoder timestep or just use it in initialization
    rnn.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
    # Training loop
    stats = defaultdict(list)
    avg_loss = 0.
    avg_rec = 0.
    avg_kl = 0.
    avg_acc = 0.
    avg_elbo = 0.
    nb_updates = 0
    print('Start training!')
    for epoch in range(nb_epochs):
        np.random.shuffle(corpus)
        for i in range(0, len(corpus), batch_size):
            rnn.zero_grad()
            X = doc.transform(corpus[i:i + batch_size])
            
            # if a column values are all the padding character for
            # for all examples, remove it.
            # because if this is not done, it causes problems beucause of the way
            # 'lengths' is used, the timestep shape of the output of the RNN
            # is determined by the maximum length in the batch.
            # if the a col is zero, the timestep shape of the output the RNN
            # will not be the same than the timestep shape of the input.
            cols = (X.sum(axis=0) != 0)
            X = X[:, cols]

            # extract the length of each sentence and sort the sentences
            # by descending order of the length
            lengths = (X != 0).sum(axis=1)
            indices = np.argsort(lengths)[::-1]
            lengths = lengths[indices]
            lengths = lengths - 1 # because inp and out have one character less than X
            lengths = lengths.tolist()
            X = X[indices]
            X = torch.from_numpy(X)
            
            # The input is : w1 w2...w_{n-1}, the target is w2 w3...w_n
            # Note that w1 = 'begin' chracter, w_n = 'end' character
            inp = X[:, 0:-1]
            assert inp.size(1) == np.max(lengths)
            inp = Variable(inp)
            inp = inp.cuda()
            unmasked_target = X[:, 1:].contiguous()
            unmasked_target = Variable(unmasked_target)
            unmasked_target = unmasked_target.cuda()
            # Pass the input through the rnn autoencoder
            unmasked_out, (latent_mean, latent_log_var, latent) = rnn(inp, lengths)
            assert unmasked_out.size(0) == inp.size(0) * inp.size(1), (unmasked_out.size(), inp.size())
            mask = (inp != 0).view(-1)
            ind = torch.range(0, mask.size(0) - 1).long()
            ind = Variable(ind)
            ind = ind.cuda()
            assert ind.size() == mask.size()
            ind = torch.masked_select(ind, mask)
            assert ind.max() < (unmasked_out.size(0) - 1)
            out = torch.index_select(unmasked_out, 0, ind)
            target = unmasked_target.view(-1)
            target = torch.index_select(target, 0, ind)
            acc = get_acc(out, target)

            kl = (-0.5 * (1 + latent_log_var - latent_mean**2 - torch.exp(latent_log_var))).sum(1).mean()
            rec = criterion(out, target)
            kl_weight = min(nb_updates / 1000., 1.)
            #loss = rec + kl_weight * (kl if kl.data[0]>10 else 0)
            #loss = rec + kl
            loss = rec + kl_weight * kl
            elbo = rec + kl
            loss.backward()
            nn.utils.clip_grad_norm(rnn.parameters(), 2)
            optimizer.step()

            stats['acc'].append(acc.data[0])
            stats['loss'].append(loss.data[0])
            stats['rec'].append(rec.data[0])
            stats['kl'].append(kl.data[0])

            avg_kl = 0.9 * avg_kl + 0.1 * kl.data[0]
            avg_rec = 0.9 * avg_rec + 0.1 * rec.data[0]
            avg_acc = 0.9 * avg_acc + 0.1 * acc.data[0]
            avg_loss = 0.9 * avg_loss + 0.1 * loss.data[0]
            avg_elbo = 0.9 * avg_elbo + 0.1 * elbo.data[0]
            if nb_updates % 100 == 0:
                print('Epoch {:03d}, [{:05d}/{:05d}], Loss : {:.3f}, acc : {:.3f}, kl : {:.3f}, recons : {:.3f}, elbo : {:.3f}'.format(epoch, i + len(X), len(corpus), avg_loss, avg_acc, avg_kl, avg_rec, avg_elbo))
                print('kl weight : {}'.format(kl_weight))
                viz.updateTrace(X=np.array([nb_updates]), Y=np.array([avg_acc]), win=win, name='acc')
                viz.updateTrace(X=np.array([nb_updates]), Y=np.array([avg_kl]), win=win, name='kl')
                viz.updateTrace(X=np.array([nb_updates]), Y=np.array([avg_rec]), win=win, name='rec')
                viz.updateTrace(X=np.array([nb_updates]), Y=np.array([avg_elbo]), win=win, name='elbo')
                viz.updateTrace(X=np.array([nb_updates]), Y=np.array([avg_loss]), win=win, name='loss')

                pd.DataFrame(stats).to_csv('{}/stats.csv'.format(folder))
                
                pred = unmasked_out.view(inp.size(0), inp.size(1), out.size(1))
                pred = pred.max(2)[1][:, :, 0]
                pred = pred.data.cpu().numpy()
                pred[pred==UNK_CHAR] = 0

                true = unmasked_target
                true = true.data.cpu().numpy()

                true = doc.inverse_transform(true)
                pred = doc.inverse_transform(pred)
                print('#### Predictions')
                for p, t in zip(pred[0:2], true[0:2]):
                    t = to_str(t)
                    p = to_str(p)
                    print('True : "{}"'.format(t))
                    print('Pred : "{}"'.format(p))
                    print('_____')
                print('### Interpolation')
                alpha = torch.linspace(0, 1, 10).view(-1, 1).repeat(1, latent_mean.size(1))
                alpha = Variable(alpha).cuda()
                l = latent_mean[0:1].repeat(alpha.size(0), 1) * alpha + latent_mean[1:2].repeat(alpha.size(0), 1) * (1 - alpha)
                out = rnn.greedy_generate(l, length)
                out = out.numpy()
                out = doc.inverse_transform(out)
                for o in out:
                    o = to_str(o)
                    print('{}'.format(o))
                
                print('### Sampling')
                l = latent_mean[0:1].repeat(10, 1)
                out = rnn.greedy_generate(l, length, deterministic=False)
                out = out.numpy()
                out = doc.inverse_transform(out)
                for o in out:
                    o = to_str(o)
                    print('Gen : {}'.format(o))

            nb_updates += 1
        torch.save(rnn, '{}/model.th'.format(folder))

def to_str(sent):
    sent = [s for s in sent if s not in (0, 1, 2, 3)]
    return ' '.join(sent)
if __name__ == '__main__':
    run(train)
