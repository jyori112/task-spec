import numpy as np
import torch
from torch import nn

def has_nan(m):
    return float(torch.isnan(m).float().mean())

def get_len_from_mask(mask):
    return mask.long().sum(-1)

class BOEDocEmbed(nn.Module):
    def forward(self, emb, mask):
        lengths = get_len_from_mask(mask).float()
        lengths = torch.max(lengths, lengths.new_ones(1))

        return torch.sum(emb, dim=1) / lengths.unsqueeze(1)

class FeedForward(nn.Module):
    def __init__(self, dims, activation, dropout):
        super().__init__()
        self.activation = activation
        self.dropout = dropout

        layers = []

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))

        self._linear_layers = nn.ModuleList(layers)

    def forward(self, x):
        output = x

        for i, layer in enumerate(self._linear_layers):
            output = self.dropout(layer(output))

            # Apply activation except for the last layer
            if i < len(self._linear_layers) - 1:
                output = self.activation(output)

        return output

class Model(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self._vocab = vocab
        self._config = config

        self._embed = nn.Embedding(self._vocab.vocabsize, config.emb_dim, padding_idx=vocab.PAD_ID)
        self.dropout = nn.Dropout(config.dropout)

        self._embed.weight.requires_grad = config.update_emb

        self.emb_dim = config.emb_dim

        if 'emb_ffnn' in config:
            dims = [self._config.emb_dim] + config.emb_ffnn.dims
            self._emb_ffnn = FeedForward(dims, torch.tanh, self.dropout)
            self.emb_dim = config.emb_ffnn.dims[-1]

        self.doc_emb_dim = self.emb_dim

    def init_network(self):
        nn.init.xavier_uniform_(self._embed.weight)

    def load_embeddings(self, fobj):
        # この関数はvocabの変化はないことを想定している。主に初期化に用いる
        for line in fobj:
            word, vec_str = line.strip().split(' ', 1)

            if word in self._vocab:
                wid = self._vocab.token2id(word)
                vec = np.fromstring(vec_str, sep=' ')
                self._embed.weight.data[wid, :] = torch.tensor(vec)

    def set_embedding(self, vocab, vecs, n_special_tokens=3):
        # この関数ではvocabの変化を仮定する。よって、単語数すらも変化して良い
        self._vocab = vocab

        # 最後の数単語はspecial tokenなので、そのままコピーする
        spec_token_emb = np.array(self._embed.weight.data[-3:].cpu())
        vecs = np.concatenate((vecs, spec_token_emb), axis=0)

        # Embedding objectを新しく作る
        self._embed = nn.Embedding(*vecs.shape)
        self._embed.weight.data[:] = torch.tensor(vecs)

    def params2update(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def embed(self, doc):
        BATCHSIZE, SEQ_SIZE = doc.shape

        emb = self._embed(doc)

        if 'emb_ffnn' in self._config:
            emb_mapped = self._emb_ffnn(torch.tanh(emb.reshape(-1, self._config.emb_dim)))
            emb = emb_mapped.reshape(BATCHSIZE, -1, self.emb_dim)

        return emb

    def forward(self, doc, mask, label=None, predict=False):
        return self._forward(doc, mask, label, predict)

    def _forward(self, doc, mask, label=None, predict=False):
        raise NotImplementedError()

    def load_state_dict_except_emb(self, state_dict, except_emb=True):
        state = self.state_dict()

        for name, param in state_dict.items():
            if except_emb and name.startswith('embed'):
                continue
            own_state[name].copy_(param)

class MultiClassModel(Model):
    def __init__(self, vocab, config):
        super().__init__(vocab, config)
        self.doc_embed = BOEDocEmbed()
        self.feedforward = FeedForward([self.doc_emb_dim, vocab.labelsize],
                                       torch.tanh, self.dropout)
        self.criterion = nn.CrossEntropyLoss()

    def init_network(self):
        super().init_network()

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def _forward(self, doc, mask, label=None, predict=False):
        emb = torch.tanh(self.embed(doc))
        doc_emb = self.doc_embed(emb, mask)
        doc_emb = self.dropout(torch.tanh(doc_emb))
        y = self.feedforward(doc_emb)

        result = {}
        result['logits'] = y.clone()

        if label is not None:
            result['loss'] = self.criterion(y, label)

        if predict:
            result['pred'] = y.argmax(dim=1)

        return result
