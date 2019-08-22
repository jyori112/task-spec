import sys
import random

import click

@click.command()
@click.option('--seed', type=int)
def main(seed):
    if seed:
        random.seed(seed)

    data = []
    for line in sys.stdin:
        data.append(line.strip())

    random.shuffle(data)

    for line in data:
        print(line)

if __name__ == '__main__':
    main()
