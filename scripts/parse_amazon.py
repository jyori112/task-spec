import sys
import json

import click

@click.command()
def main():
    sys.stdin.readline()
    for line in sys.stdin:
        line = line.split('\t')
        score, text = line[7], line[13]

        score = float(score)

        if score < 3:
            label = 'NEGATIVE'
        elif score > 3:
            label = 'POSITIVE'
        else:
            label = 'NEUTRAL'

        text = text.replace('<br />', ' ').replace('\n', ' ')

        print(json.dumps(dict(label=label, text=text)))

if __name__ == '__main__':
    main()
