import json
import sys

import click

@click.command()
def main():
    for line in sys.stdin:
        data = json.loads(line)

        if data['stars'] < 3:
            label = 'NEGATIVE'
        elif data['stars'] > 3:
            label = 'POSITIVE'
        else:
            label = 'NEUTRAL'

        text = data['text'].replace('\n', ' ')

        print(json.dumps(dict(label=label, text=text)))

if __name__ == '__main__':
    main()
