import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import click
import numpy as np
import torch
from torch import optim
from torch import autograd

from data import Vocab, Dataset
from model import MultiClassModel
from config import Config

def compute_accuracy(x, y):
    return torch.mean((x == y).float())

def evaluate(model, dataset, report, device):
    model.eval()
    accuracies = []
    losses = []
    for labels, tokens, mask in dataset.batches():
        labels = labels.to(device)
        tokens = tokens.to(device)
        mask = mask.to(device)
        result = model(tokens, mask, label=labels, predict=True)
        losses.append(float(result['loss']))
        accuracy = compute_accuracy(result['pred'], labels)
        accuracies.append(float(accuracy))

    report['dev_loss'] = np.mean(losses)
    report['dev_accuracy'] = np.mean(accuracies)
    model.train()

def has_nan(m):
    return torch.isnan(m).float().sum() > 0

@click.command()
@click.argument('config', type=click.Path(exists=True))
@click.argument('model-dir', type=click.Path())
@click.option('--overwrite/--no-overwrite', '-f', default=False)
@click.option('--seed', type=int, default=1)
def main(config, model_dir, overwrite, seed):
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    config = Config(config)

    model_dir = Path(model_dir)

    if model_dir.exists() and not overwrite:
        raise Exception('Output dir `{}` already exists'.format(model_dir))

    if not model_dir.exists():
        os.mkdir(model_dir)

    config.save(model_dir / 'config.json')

    print("Load vocab...", end='', file=sys.stderr, flush=True)
    with open(config.vocab.words) as fwords, open(config.vocab.labels) as flabels:
        vocab = Vocab.load(fwords, flabels)
    print("Done", file=sys.stderr, flush=True)


    print("Prepare dataset...", end='', file=sys.stderr, flush=True)
    train_set = Dataset(config.train_set, vocab)
    dev_set = Dataset(config.dev_set, vocab)
    print("Done", file=sys.stderr, flush=True)

    print("Create model...", end='', file=sys.stderr, flush=True)
    model = MultiClassModel(vocab, config.model)
    print("Done", file=sys.stderr, flush=True)

    print("Initialize model...", end='', file=sys.stderr, flush=True)
    model.init_network()
    print("Done", file=sys.stderr, flush=True)

    if 'emb_path' in config.model:
        print("Load embeddings...", end='', file=sys.stderr, flush=True)
        with open(config.model.emb_path) as f:
            model.load_embeddings(f)
        print("Done", file=sys.stderr, flush=True)

    print("Sending model to GPU...", end='', file=sys.stderr, flush=True)
    device = torch.device(config.training.device)
    model = model.to(device)
    print("Done", flush=True)

    #opt = optim.SGD(model.params2update(), lr=0.001)
    opt = optim.Adam(model.params2update())

    train_losses = []
    train_accuracy = []

    for epoch in range(config.training.num_epoch):
        report = {}
        report['epoch'] = epoch
        report['time'] = datetime.now().isoformat()

        epoch_start = time.time()
        for step, (label, docs, mask) in enumerate(train_set.batches(config.training.batchsize)):
            #print(step, label, docs, mask)
            label = label.to(device)
            docs = docs.to(device)
            mask = mask.to(device)

            with autograd.detect_anomaly():
                result = model.forward(docs, mask, label=label, predict=True)

                if torch.isnan(result['loss']).any():
                    raise Exception("Loss is NaN at step {} of epoch {}".format(step, epoch))

                opt.zero_grad()
                result['loss'].backward()

                for name, param in model.named_parameters():
                    if param.requires_grad and has_nan(param.grad):
                        #print(param.grad)
                        param.grad.data[torch.isnan(param.grad)] = 0
                        print("NAN", name)

                opt.step()

            train_losses.append(float(result['loss']))
            train_accuracy.append(float(compute_accuracy(label, result['pred'])))

            if step % 1000 == 0:
                print('{}\t{}'.format(epoch, step))


        epoch_end = time.time()

        report['epoch_duration'] = epoch_end - epoch_start
        report['train_loss'] = np.mean(train_losses)
        report['train_accuracy'] = np.mean(train_accuracy)
        evaluate(model, dev_set, report, device)

        with open(model_dir / 'log.jsonl', 'a') as f:
            print(json.dumps(report), file=f)

        torch.save(model.state_dict(), model_dir / 'ckpt.epoch-{}.model'.format(epoch))

if __name__ == '__main__':
    main()
