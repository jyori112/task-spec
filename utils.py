import logging

import numpy as np
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Vocab

logger = logging.getLogger(__name__)

def load_emb(fp):
    n_vocab, dim = map(int, fp.readline().split())

    emb = KeyedVectors(dim)
    emb.vectors = np.empty((n_vocab, dim), dtype=np.float32)

    for i, line in enumerate(fp):
        word, vec_str = line.split(' ', 1)
        emb.vectors[i] = np.fromstring(vec_str, sep=' ')
        emb.vocab[word] = Vocab(index=i)
        emb.index2word.append(word)

    return emb

def batchfy(iterators, batchsize):
    buf = []

    for items in zip(*iterators):
        buf.append(items)

        if len(buf) >= batchsize:
            yield tuple(zip(*buf))
            buf = []

    yield tuple(zip(*buf))

