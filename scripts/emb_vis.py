import sys
import json

import click
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import embeddings

@click.command()
@click.argument('config', type=click.Path())
@click.argument('output', type=click.Path())
def main(config, output):
    with open(config) as f:
        config = json.load(f)

    n_vocab = config['n_vocab']

    categories = config['categories']

    for cat in categories:
        if 'path' in categories[cat]:
            with open(categories[cat]['path']) as f:
                categories[cat]['words'] = list(set(f.read().lower().strip().split()))

    # Load Embeddings
    print("Load embeddings", file=sys.stderr)
    words, vecs = embeddings.load_emb(sys.stdin, n_vocab)
    word2id = {word: idx for idx, word in enumerate(words)}

    vis_word_idx = []
    start_idx = {}
    end_idx = {}

    for cat in categories:
        start_idx[cat] = len(vis_word_idx)
        vis_word_idx += [word2id[word] for word in categories[cat]['words'] if word in word2id]
        end_idx[cat] = len(vis_word_idx)

    print("Applying t-SNE", file=sys.stderr)
    coord = TSNE(n_components=2).fit_transform(vecs[vis_word_idx])
    print("Done", file=sys.stderr)

    plt.figure(figsize=(10, 10))
    plt.xticks([], [])
    plt.yticks([], [])

    print("Plotting", file=sys.stderr)
    for cat, opt in categories.items():
        plt.plot(coord[start_idx[cat]:end_idx[cat], 0], coord[start_idx[cat]:end_idx[cat], 1], opt['marker'])

    plt.savefig(output, bbox_inches='tight')

if __name__ == '__main__':
    main()
