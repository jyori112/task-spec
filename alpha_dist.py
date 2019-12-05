import logging

import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

@click.group()
def cli():
    pass

def load_data(path, absolute=False):
    # Load data
    weights = np.load(path)

    if absolute:
        weights = np.abs(weights)

    n_vocab = weights.shape[0]
    k = weights.shape[1]

    return weights, (n_vocab, k)

@cli.command()
@click.argument('weight-path', type=click.Path(exists=True))
@click.argument('output-path', type=click.Path())
@click.option('--absolute/--no-abs', default=False)
@click.option('--ymin', type=float)
@click.option('--ymax', type=float)
def violin(weight_path, output_path, absolute, ymin, ymax):
    weights, (n_vocab, k) = load_data(weight_path, absolute)

    ids = np.empty(shape=weights.shape, dtype=np.int32)

    for i in range(k):
        ids[:, i] = i

    pd_weights = pd.DataFrame({
        'weights': weights.reshape(n_vocab * k),
        'i': ids.reshape(n_vocab * k)})
    
    sns.violinplot(x='i', y='weights', data=pd_weights)

    plt.ylim(ymin, ymax)
    plt.savefig(output_path)

@cli.command()
@click.argument('weight-path', type=click.Path(exists=True))
@click.argument('output-path', type=click.Path())
@click.option('--absolute/--no-abs', default=False)
@click.option('--ymin', type=float)
@click.option('--ymax', type=float)
def violin_top_vs_else(weight_path, output_path, absolute, ymin, ymax):
    weights, (n_vocab, k) = load_data(weight_path, absolute)

    ids = np.empty(shape=weights.shape, dtype=np.int32)

    for i in range(k):
        ids[:, i] = i

    pd_weights = pd.DataFrame({
        'weights': weights.reshape(n_vocab * k),
        'i': ids.reshape(n_vocab * k)})
    
    sns.violinplot(x='i', y='weights', data=pd_weights)

    plt.ylim(ymin, ymax)
    plt.savefig(output_path)

@cli.command()
@click.argument('weight-path', type=click.Path(exists=True))
@click.argument('output-path', type=click.Path())
@click.option('--absolute/--no-abs', default=False)
@click.option('--xmin', type=float)
@click.option('--xmax', type=float)
@click.option('--ymin', type=float)
@click.option('--ymax', type=float)
def kde(weight_path, output_path, absolute, xmin, xmax, ymin, ymax):
    weights, (n_vocab, k) = load_data(weight_path, absolute)

    for i in range(k):
        sns.kdeplot(weights[:, i], label=i+1)

    plt.legend()

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.savefig(output_path)

@cli.command()
@click.argument('weight-path', type=click.Path(exists=True))
@click.argument('output-path', type=click.Path())
@click.option('--mean-normalize/--no-mean-normalize', default=False)
@click.option('--log/--no-log', default=False)
@click.option('--absolute/--no-abs', default=False)
@click.option('--xmin', type=float)
@click.option('--xmax', type=float)
@click.option('--ymin', type=float)
@click.option('--ymax', type=float)
def fit_slope(weight_path, output_path, absolute, mean_normalize, log, xmin, xmax, ymin, ymax):
    weights, (n_vocab, k) = load_data(weight_path, absolute)

    ids = np.empty(shape=weights.shape, dtype=np.int32)

    for i in range(k):
        ids[:, i] = i

    if log:
        weights = np.log(weights-np.min(weights)+0.001)

    weights = weights - np.mean(weights, axis=1)[:, None]
    ids = ids - np.mean(ids, axis=1)[:, None]

    a = np.sum(weights * ids, axis=1) / np.sum(ids ** 2, axis=1)

    sns.kdeplot(a)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.savefig(output_path)

@cli.command()
@click.argument('weight-path', type=click.Path(exists=True))
@click.argument('output-path', type=click.Path())
@click.option('--absolute/--no-abs', default=False)
@click.option('--xmin', type=float)
@click.option('--xmax', type=float)
@click.option('--ymin', type=float)
@click.option('--ymax', type=float)
def exponential(weight_path, output_path, absolute, xmin, xmax, ymin, ymax):
    weights, (n_vocab, k) = load_data(weight_path, absolute)

    ids = np.empty(shape=weights.shape, dtype=np.int32)

    for i in range(k):
        ids[:, i] = i

    a = np.sum(weights * ids, axis=1) / np.sum(ids ** 2, axis=1)

    sns.kdeplot(a)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.savefig(output_path)



@cli.command()
@click.argument('weight-path', type=click.Path(exists=True))
@click.argument('output-path', type=click.Path())
@click.option('--absolute/--no-abs', default=False)
@click.option('--xmin', type=float)
@click.option('--xmax', type=float)
@click.option('--ymin', type=float)
@click.option('--ymax', type=float)
def top2diff(weight_path, output_path, absolute, xmin, xmax, ymin, ymax):
    weights, (n_vocab, k) = load_data(weight_path, absolute)

    sns.kdeplot(weights[0] - weights[1])

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.savefig(output_path)

if __name__ == '__main__':
    cli()

