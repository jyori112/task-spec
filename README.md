This repository hosts codes to reproduce the results in the paper "Multilingual model using cross-task embedding projection" published at CoNLL 2019.
For the implementation of Locally Linear Mapping, please check out https://github.com/jyori112/llm.

# Multilingual model using cross-task embedding projection

## Setup

Clone this repository in to your system:
```
$ git clone install https://github.com/jyori112/task-spec
$ cd task-spec
```

Install python packages:
```
$ pip install -r requirements.txt
```

## Usage

### Obtain cross-lingual word embeddings

We recommend to utilize [VecMap tool](https://github.com/artetxem/vecmap) to obtain cross-lingual word embeddings.

### Train a neural network model

To train a model, you need to create a config file that contains all the information required for training.
The config file should look like
```
{
    "train_set": "[PATH_TO_TRAINING_SET]",
    "dev_set": "[PATH_TO_DEV_SET]",
    "vocab": {
        "words": "[PATH_TO_VOCAB_FILE]",
        "labels": "[PATH_TO_LABEL_FILE]"
    },
    "model": {
        "emb_dim": 300,
        "dropout": 0.2,
        "update_emb": true,
        "emb_path": "[INITIAL_EMBEDDING_PATH]"
    },
    "training": {
        "num_epoch": 20,
        "batchsize": 32,
        "device": "cuda"
    }
}
```

To train a model (including the embedding layer),

```
$ python -m taskspec.train [CONFIG_FILE] [MODEL_DIR]
```

This script assumes that the training/development datasets are given in jsonl format.
Each line contains the json representation of an example with keys, `label` and `text`.

To fine the best checkpoint, run

```
$ cat [MODEL_DIR]/log.jsonl| jq '[.epoch,.dev_accuracy]|@tsv' -r | sort -rnk2 | head -n1 | cut -f1
```

### Extract embedding of a trained model

To extract the embedding of a trained model,

```
$ python -m taskspec.extract_emb [MODEL_DIR] --checkpoint [CHECKPOINT]
```

This will output the embedding layer to the `stdout`.

### Apply locally linear mapping

Locally linear mapping is hosted at https://github.com/jyori112/llm.
To clone this project, simply run

```
$ git clone https://github.com/jyori112/llm
```

Please refer to the repository for the usage of locally linear mapping.

### Evaluate the model

To evaluate a model, run

```
$ python -m taskspec.evalaute [MODEL_DIR] [EVALUATION_PATH] --emb-path [EMB_PATH]
```

`[EVALUATION_PATH]` should be the same format as the training data.

## Citation

If you use this code for research, please cite the following paper:

```
@inproceedings{sakuma2019conll,
    author = {Sakuma, Jin and Yoshinaga, Naoki},
    title = {Multilingual Model Using Cross-Task Embedding Projection},
    booktitle = {Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL)},
    year = {2019},
    pages = {22--32}
}
```