import sys

import click

from data import Vocab

@click.command()
@click.option('--min-count', type=int, default=5)
@click.option('--word-path', type=click.Path())
@click.option('--label-path', type=click.Path())
def main(min_count, word_path, label_path):
    vocab = Vocab.build(sys.stdin, min_count)

    with open(word_path, 'w') as word_file, open(label_path, 'w') as label_file:
        vocab.save(word_file, label_file)

if __name__ == '__main__':
    main()
