from pathlib import Path

import click
import numpy as np
import torch

from taskspec.data import Vocab
from taskspec.model import MultiClassModel
from taskspec.utils import Config

def compute_accuracy(x, y):
    return torch.mean((x == y).float())

@click.command()
@click.argument('model-dir', type=click.Path(exists=True))
@click.option('--ckpt', type=str, default='best.model')
def main(model_dir, ckpt):
    model_dir = Path(model_dir)
    config = Config(model_dir / 'config.json')

    with open(config.vocab.words) as fwords, open(config.vocab.labels) as flabels:
        vocab = Vocab.load(fwords, flabels)

    model = MultiClassModel(vocab, config.model)
    model.load_state_dict(torch.load(model_dir / ckpt, map_location='cpu'))

    vecs = np.array(model._embed.weight.data)

    print('{} {}'.format(vocab.vocabsize, vecs.shape[1]))

    for i in range(vocab.vocabsize):
        word = vocab.id2token(i)
        vec_str = ' '.join('{:.6f}'.format(v) for v in vecs[i])
        print("{} {}".format(word, vec_str))

if __name__ == '__main__':
    main()
