import sys
import logging
from collections import defaultdict

import click
import numpy as np
import cupy
from tqdm import tqdm

import embeddings

logger = logging.getLogger(__name__)

@click.group()
def cli():
    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

@cli.command()
def variance():
    data = []
    for line in sys.stdin:
        word, inv_nns = line.split('\t')
        n_inv_nns = len(inv_nns.split())
        data.append(n_inv_nns)

    print(np.var(data))

if __name__ == '__main__':
    cli()
