import json

from collections import Counter

import numpy as np

import torch

SOS = '__SOS__'
EOS = '__EOS__'
PAD = '__PAD__'

class Vocab:
    def __init__(self, words, labels):
        self.vocabsize = len(words)
        self.labelsize = len(labels)

        self._id2word = {wid: word for wid, word in enumerate(words)}
        self._word2id = {word: wid for wid, word in self._id2word.items()}

        self._id2label = {wid: label for wid, label in enumerate(labels)}
        self._label2id = {label: wid for wid, label in self._id2label.items()}

        self.PAD_ID = self._word2id[PAD]

    def __contains__(self, word):
        return word in self._word2id

    def token2id(self, token):
        if token in self._word2id:
            return self._word2id[token]
        else:
            return self._word2id[PAD]

    def id2token(self, wid):
        return self._id2word[wid]

    def label2id(self, label):
        return self._label2id[label]

    def id2label(self, lid):
        return self._id2label[lid]

    def words(self):
        for i in range(self.vocabsize):
            yield self.id2token(i)

    def idfy(self, tokens):
        if isinstance(tokens, str):
            tokens = tokens.split()

        return [self.token2id(token) for token in tokens]

    def batch_idfy(self, batch_tokens, pad=True):
        batch_ids = [self.idfy(tokens) for tokens in batch_tokens]

        if not pad:
            return batch_ids

        lengths = [len(ids) for ids in batch_ids]
        max_len = max(lengths)
        batch_ids = [idx + [self.PAD_ID] * (max_len - len(idx)) for idx in batch_ids]
        batch_ids = np.array(batch_ids, dtype=np.int32)
        mask = (batch_ids != self.PAD_ID).astype(np.int32)

        return batch_ids, mask

    def save(self, word_file, label_file):
        for wid in range(self.vocabsize):
            print(self._id2word[wid], file=word_file)

        for lid in range(self.labelsize):
            print(self._id2label[lid], file=label_file)

    @staticmethod
    def load(word_file, label_file):
        words = word_file.read().strip().split()
        labels = label_file.read().strip().split()

        return Vocab(words, labels)

    @staticmethod
    def build(fobj, min_count=5, token_key='text', label_key='label'):
        word_counter = Counter()
        labels = set()

        for line in fobj:
            data = json.loads(line)
            tokens = data[token_key].split()
            label = data[label_key]

            word_counter.update(tokens)
            labels.add(label)

        words = [(word, count) for word, count in word_counter.items() if count >= min_count]
        words = sorted(words, key=lambda x: x[1], reverse=True)
        words = [word for word, count in words] + [EOS, SOS, PAD]

        return Vocab(words, sorted(list(labels)))

    @staticmethod
    def build_from_emb(emb_file, label_file):
        labels = label_file.read().strip().split()

        n_vocab, emb_dim = map(int, emb_file.readline().strip().split())

        words = []
        vecs = np.empty((n_vocab, emb_dim), dtype=np.float32)

        for i, line in enumerate(emb_file):
            word, vec_str = line.split(' ', 1)
            words.append(word)
            vec = np.fromstring(vec_str, sep=' ')
            vecs[i] = vec

        words += [EOS, SOS, PAD]

        # この時点で、special tokenのembeddingは設定されていないことに注意（embeddingの数が小さい）
        return Vocab(words, labels), vecs


class Dataset:
    def __init__(self, path, vocab, max_len=None, shuffle=True):
        self._path = path
        self._vocab = vocab
        self._max_len = max_len
        self._shuffle = shuffle
        self._labels = []
        self._documents = []
        self._load_complete = False

    def batch_pad(self, batch_tokens):
        max_len = max(len(tokens) for tokens in batch_tokens)
        output = np.full((len(batch_tokens), max_len), self._vocab.PAD_ID)

        for i, tokens in enumerate(batch_tokens):
            output[i, :len(tokens)] = tokens

        return output, (output != self._vocab.PAD_ID).astype(np.int32)

    def load_and_batch(self, batchsize=32):
        with open(self._path) as f:
            batch_tokens = []
            batch_label = []

            for line in f:
                line = json.loads(line)
                label = line['label']
                tokens = line['text'].split()

                # To ids
                lid = self._vocab.label2id(label)
                tokens = self._vocab.idfy(tokens)

                # Save data
                self._labels.append(lid)
                self._documents.append(tokens)

                # Add to batch
                batch_label.append(lid)
                batch_tokens.append(tokens)

                if len(batch_label) == batchsize:
                    batch_labels = torch.tensor(np.array(batch_label), dtype=torch.long)
                    batch_padded_tokens, batch_mask = self.batch_pad(batch_tokens)
                    batch_padded_tokens = torch.tensor(batch_padded_tokens, dtype=torch.long)
                    batch_mask = torch.tensor(batch_mask)

                    yield batch_labels, batch_padded_tokens, batch_mask

                    batch_label, batch_tokens = [], []

        self._labels = np.array(self._labels)
        self._documents = np.array(self._documents)
        self.datasize = len(self._labels)
        self._load_complete = True

    def batches(self, batchsize=32):
        if self._load_complete:
            # the data is already loaded. shuffle the data to create batches
            if self._shuffle:
                order = np.random.permutation(self.datasize)
            else:
                order = np.arange(self.datasize)

            for index in range(0, self.datasize, batchsize):
                indexes = order[index:min(self.datasize, index+batchsize)]
                batch_labels = torch.tensor(self._labels[indexes], dtype=torch.long)
                batch_tokens = [self._documents[i] for i in indexes]
                batch_padded_tokens, batch_mask = self.batch_pad(batch_tokens)
                batch_padded_tokens = torch.tensor(batch_padded_tokens, dtype=torch.long)
                batch_mask = torch.tensor(batch_mask)

                yield batch_labels, batch_padded_tokens, batch_mask

        else:
            for batch in self.load_and_batch(batchsize=batchsize):
                yield batch
