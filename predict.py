import sys
import json
from pathlib import Path

import click
import numpy as np
import torch

from data import Vocab, Dataset
from model import MultiClassModel
from config import Config

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

def batch_pad(batch_tokens, vocab):
    max_len = max(len(tokens) for tokens in batch_tokens)
    output = np.full((len(batch_tokens), max_len), vocab.PAD_ID)

    for i, tokens in enumerate(batch_tokens):
        output[i, :len(tokens)] = tokens

    return output, (output != vocab.PAD_ID).astype(np.int32)

@click.command()
@click.argument('model-dir', type=click.Path(exists=True))
@click.option('--ckpt', type=str, default='best.model')
@click.option('--device', type=str)
@click.option('--emb-path', type=click.Path(exists=True))
def main(model_dir, ckpt, device, emb_path):
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

    input_lines = []
    batch_tokens = []
    for i, line in enumerate(sys.stdin):
        input_lines.append(line)
        tokens = line.split()
        tokens = vocab.idfy(tokens)
        batch_tokens.append(tokens)

        if (i + 1) % 32 == 0:
            batch_padded_tokens, batch_mask = batch_pad(batch_tokens, vocab)
            batch_padded_tokens = torch.tensor(batch_padded_tokens, dtype=torch.long).to(device)
            batch_mask = torch.tensor(batch_mask).to(device)
            result = model(batch_padded_tokens, batch_mask, predict=True)

            for i in range(len(result['pred'])):
                print(json.dumps({
                    'input': input_lines[i], 
                    'prediction': vocab.id2label(int(result['pred'][i]))
                }))

    batch_padded_tokens, batch_mask = batch_pad(batch_tokens, vocab)
    result = model(batch_padded_tokens, batch_mask, predict=True)

    for i in len(result['pred']):
        print(json.dumps({
            'input': input_lines[i], 
            'prediction': vocab.id2label(int(result['pred'][i]))
        }))

if __name__ == '__main__':
    main()
