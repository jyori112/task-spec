import logging
import json

import numpy as np
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Vocab
from tqdm import tqdm

logger = logging.getLogger(__name__)

def load_emb(fp):
    n_vocab, dim = map(int, fp.readline().split())

    emb = KeyedVectors(dim)
    emb.vectors = np.empty((n_vocab, dim), dtype=np.float32)

    for i, line in tqdm(enumerate(fp), total=n_vocab, unit='word'):
        word, vec_str = line.split(' ', 1)
        emb.vectors[i] = np.fromstring(vec_str, sep=' ')
        emb.vocab[word] = Vocab(index=i)
        emb.index2word.append(word)

    return emb

def batchfy(iterator, batchsize):
    buf = []

    for item in iterator:
        buf.append(item)

        if len(buf) >= batchsize:
            yield buf
            buf = []

    if buf:
        yield buf


def batchfy_old(iterators, batchsize):
    buf = []

    for items in zip(*iterators):
        buf.append(items)

        if len(buf) >= batchsize:
            yield tuple(zip(*buf))
            buf = []

    yield tuple(zip(*buf))

class ConfigNotFoundError(Exception):
    def __init__(self, node_name, name):
        full_name = '{}.{}'.format(node_name, name)


        super().__init__(self, 'Configuration `{}` does not exists'.format(full_name))

class ConfigNode:
    def __init__(self, name, data):
        self._name = name
        self._data = data

    def __getattr__(self, name):
        if name not in self._data:
            raise ConfigNotFoundError(self._name, name)

        ret = self._data[name]

        if isinstance(ret, dict):
            return ConfigNode('{}.{}'.format(self._name, name), ret)

        return ret

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __contains__(self, name):
        return name in self._data

class Config(ConfigNode):
    def __init__(self, path):
        self.path = path

        with open(path) as f:
            data = json.load(f)

        super().__init__('', data)

    def save(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self._data, f, indent=4)
