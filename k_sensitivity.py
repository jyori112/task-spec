import sys
from collections import defaultdict
from itertools import product

import click
import numpy as np
import matplotlib.pyplot as plt

@click.command()
@click.argument('output', type=click.Path())
def main(output):
    results = defaultdict(lambda: defaultdict(list))
    for line in sys.stdin:
        lang, trial, k, dev_accuracy, accuracy = line.strip().split('\t')
        k = int(k)
        accuracy = float(accuracy)

        results[lang][k].append(accuracy)

    ks = list(range(1,11))
    line_style = ['-', '--', '-.', ':']
    marker = ['o', 'x', '+']

    plt_style = product(line_style, marker)

    for lang, accuracies in results.items():
        style = ''.join(next(plt_style))
        plt.plot(ks, [np.mean(accuracies[k]) for k in ks], style, label=lang)

    plt.ylim(0.3, 1)
    plt.xticks(ks, ks, fontsize=13)
    plt.yticks(fontsize=13)

    plt.xlabel('k', fontsize=15)
    plt.ylabel('accuracy', fontsize=15)

    plt.legend(ncol=4, fontsize=15, loc='lower left')

    plt.savefig(output, bbox_inches='tight')

if __name__ == '__main__':
    main()
