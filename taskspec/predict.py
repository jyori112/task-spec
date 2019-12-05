import sys
import json
from pathlib import Path
import logging
import gc

import click
import numpy as np
import torch

from data import Vocab, Dataset
from model import MultiClassModel
from config import Config

logger = logging.getLogger(__name__)
LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

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
    logger.info("Loading parameters")
    model.load_state_dict(torch.load(model_dir / ckpt))

    if emb_path:
        logger.info("Loading embeddings")
        with open(emb_path) as femb, open(config.vocab.labels) as flabels:
            vocab, vecs = Vocab.build_from_emb(femb, flabels)

        model.set_embedding(vocab, vecs)

    if device is not None:
        device = torch.device(device)
    else:
        device = torch.device(config.training.device)

    model = model.to(device)

    model.eval()

    logger.info("Start prediction")
    input_lines = []
    batch_tokens = []
    for i, line in enumerate(sys.stdin):
        input_lines.append(line)
        tokens = line.split()
        tokens = vocab.idfy(tokens)
        batch_tokens.append(tokens)

        if (i + 1) % 32 == 0:
            batch_padded_tokens, batch_mask = batch_pad(batch_tokens, vocab)
            batch_padded_tokens = torch.tensor(batch_padded_tokens, dtype=torch.long, requires_grad=False).to(device)
            batch_mask = torch.tensor(batch_mask, requires_grad=False).to(device)
            result = model(batch_padded_tokens, batch_mask, predict=True)

            prob = torch.nn.functional.softmax(result['logits'], dim=1)

            for i in range(len(result['pred'])):
                print(json.dumps({
                    'input': input_lines[i], 
                    'prob': {vocab.id2label(j): float(prob[i,j]) for j in range(prob.shape[1])},
                    'prediction': vocab.id2label(int(result['pred'][i]))
                }))
            input_lines = []
            batch_tokens = []

    if not input_lines:
        return

    batch_padded_tokens, batch_mask = batch_pad(batch_tokens, vocab)
    result = model(batch_padded_tokens, batch_mask, predict=True)

    prob = torch.nn.functional.softmax(result['logits'], dim=1)

    for i in len(result['pred']):
        print(json.dumps({
            'input': input_lines[i], 
            'prob': {vocab.id2label(j): float(prob[i,j]) for j in range(prob.shape[1])},
            'prediction': vocab.id2label(int(result['pred'][i]))
        }))

if __name__ == '__main__':
    main()
