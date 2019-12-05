import json
from pathlib import Path
import logging

import click
import numpy as np
import torch
import cupy

from taskspec.data import Vocab, Dataset
from taskspec.model import MultiClassModel
from taskspec.utils import Config

logger = logging.getLogger(__name__)
LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

def compute_accuracy(x, y):
    return torch.mean((x == y).float())


def evaluate(model, eval_set, device):
    model.eval()
    accuracies = []
    losses = []

    for labels, tokens, mask in eval_set.batches():
        labels = labels.to(device)
        tokens = tokens.to(device)
        mask = mask.to(device)
        result = model(tokens, mask, label=labels, predict=True)
        losses.append(float(result['loss']))
        accuracy = compute_accuracy(result['pred'], labels)
        accuracies.append(float(accuracy))

    return np.mean(losses), np.mean(accuracies)

@click.command()
@click.argument('model-dir', type=click.Path(exists=True))
@click.argument('eval-set-path', type=click.Path(exists=True))
@click.option('--dev-path', type=click.Path(exists=True))
@click.option('--ckpt', type=str, default='best.model')
@click.option('--device', type=str)
@click.option('--emb-path', type=click.Path(exists=True))
def main(model_dir, eval_set_path, dev_path, ckpt, device, emb_path):
    model_dir = Path(model_dir)
    config = Config(model_dir / 'config.json')

    with open(config.vocab.words) as fwords, open(config.vocab.labels) as flabels:
        vocab = Vocab.load(fwords, flabels)

    model = MultiClassModel(vocab, config.model)
    model.load_state_dict(torch.load(model_dir / ckpt))

    if emb_path:
        with open(emb_path) as femb, open(config.vocab.labels) as flabels:
            vocab, vecs = Vocab.build_from_emb(femb, flabels)

        model.set_embedding(vocab, vecs)

    if device is not None:
        device = torch.device(device)
    else:
        device = torch.device(config.training.device)

    model = model.to(device)

    eval_set = Dataset(eval_set_path, vocab)

    report = {}
    report['loss'], report['accuracy'] = evaluate(model, eval_set, device)
    report['eval_set'] = eval_set_path

    if dev_path is not None:
        dev_set = Dataset(dev_path, vocab)
        report['dev_loss'], report['dev_accuracy'] = evaluate(model, dev_set, device)
        report['dev_set'] = dev_path

    report['ckpt'] = ckpt

    if emb_path is not None:
        report['emb_path'] = emb_path

    print(json.dumps(report))

if __name__ == '__main__':
    main()
