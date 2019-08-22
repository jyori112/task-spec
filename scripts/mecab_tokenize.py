import sys

import click
import MeCab

tagger = MeCab.Tagger('-Owakati')

@click.command()
def main():
    for line in sys.stdin:
        sentences = line.split('<br />')
        tokens = ''
        for sentence in sentences:
            tokens += tagger.parse(sentence).strip()
        print(tokens)

if __name__ == '__main__':
    main()
