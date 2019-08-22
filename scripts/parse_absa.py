import sys
import json

import xmltodict

def main():
    data = xmltodict.parse(sys.stdin.read())

    for review in data['Reviews']['Review']:
        for sent in review['sentences']['sentence']:
            if 'Opinions' not in sent:
                continue

            if isinstance(sent, str):
                continue

            if sent['Opinions'] is None:
                continue

            opinions = sent['Opinions']['Opinion']

            if not isinstance(opinions, list):
                opinions = [opinions]

            text = sent['text']
            polarities = list(set(opinion['@polarity'] for opinion in opinions))

            if len(polarities) != 1:
                continue

            polarity = polarities[0].upper()
            print(json.dumps(dict(label=polarity, text=text)))

if __name__ == '__main__':
    main()
