import sys
from pathlib import Path
import json

import xmltodict
import click

def tolist(obj):
    if not isinstance(obj, list):
        return [obj]

    return obj

@click.command()
@click.argument('dirpath', type=click.Path(exists=True))
def main(dirpath):
    for filepath in Path(dirpath).glob('**/*.xml'):
        with open(filepath) as fobj:
            try:
                xml = fobj.read()
                doc = xmltodict.parse(xml)

                text = ' '.join(p for p in tolist(doc['newsitem']['text']['p']) if p is not None)

                codes = tolist(doc['newsitem']['metadata']['codes'])

                topics = []

                for code in codes:
                    if code['@class'] == 'bip:topics:1.0':
                        for code_class in tolist(code['code']):
                            topic = code_class['@code']
                            if topic in ('CCAT', 'ECAT', 'GCAT', 'MCAT'):
                                topics.append(topic)

                if len(set(topics)) != 1:
                    continue

                print(json.dumps(dict(label=topics[0], text=text)))
            except:
                print("Error {}".format(sys.exc_info()[0]), file=sys.stderr)

if __name__ == '__main__':
    main()
