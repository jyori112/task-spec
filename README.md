## Requirements

```
1. mecab with ipadic dictionary
2. jq
```

## Directory Structure

```
data/
  - orig/
  - interim/
    - datasets/
      - yelp/
        - en.labels.txt
        - en.tokenized.txt
        - en.all.jsonl
      - amazon/
        - {lang}.labels.txt
        - {lang}.tokenized.txt
        - {lang}.all.jsonl
      - rcv/
        - {lang}.parsed.jsonl
        - {lang}.labels.txt
        - {lang}.tokenized.txt
        - {lang}.all.jsonl
      - absa/
        - {lang}.parsed.jsonl
        - {lang}.labels.txt
        - {lang}.tokenized.txt
        - {lang}.all.jsonl
  - processed/
    - datasets/
      - yelp/
        - en.train.jsonl
        - en.test.jsonl
        - en.dev.jsonl
      - amazon/
        - en.train.jsonl
        - en.test.jsonl
        - en.dev.jsonl
        - {lang}.test.jsonl
        - {lang}.dev100.jsonl
      - rcv/
        - en.train.jsonl
        - en.test.jsonl
        - en.dev.jsonl
        - {lang}.test.jsonl
        - {lang}.dev100.jsonl
      - absa/
        - {lang}.test.jsonl
        - {lang}.dev100.jsonl
    - wordemb/
      - wiki.{lang}.vec
    - clwe/
      - en-{lang}/
        - log.tsv
        - en.vec
        - {lang}.vec
    - models/
      - amazon/
        - crosstask{1,2,3}
        - genemb{1,2,3}
        - embffnn{1,2,3}
        - en-{lang}.genemb{1,2,3}/
        - en-{lang}.embffnn{1,2,3}/
      - yelp/
        - crosstask{1,2,3}
        - genemb{1,2,3}
        - embffnn{1,2,3}
        - en-{lang}.genemb{1,2,3}/
        - en-{lang}.embffnn{1,2,3}/
      - rcv/
        - crosstask{1,2,3}
        - genemb{1,2,3}
        - embffnn{1,2,3}
        - en-{lang}.genemb{1,2,3}/
        - en-{lang}.embffnn{1,2,3}/
```
