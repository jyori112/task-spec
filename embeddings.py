import sys
import logging
from collections import defaultdict

import click
import numpy as np
import cupy
from tqdm import tqdm

logger = logging.getLogger(__name__)

def load_emb(fobj, n_vocab=None):
    _n_vocab, dim = map(int, fobj.readline().split())

    n_vocab = n_vocab or _n_vocab

    vecs = np.empty((n_vocab, dim), dtype=np.float32)

    words = []

    for i, line in enumerate(fobj):
        if i >= n_vocab:
            break

        word, vec_str = line.split(' ', 1)
        words.append(word)
        vecs[i] = np.fromstring(vec_str, sep=' ')

    return words, vecs

def save_emb(fobj, words, vecs):
    print("{} {}".format(len(words), len(vecs[0])), file=fobj)

    for i, word in enumerate(words):
        vec_str = ' '.join('{:.6f}'.format(v) for v in vecs[i])
        print('{} {}'.format(word, vec_str), file=fobj)

def normalize_emb(vecs):
    return vecs / np.linalg.norm(vecs, axis=1)[:, None]

def get_word2id(words):
    return {word: idx for idx, word in enumerate(words)}

@click.group()
def cli():
    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

@cli.command()
@click.argument('emb-path', type=click.Path(exists=True))
@click.option('--k', type=int, default=10)
@click.option('--n-vocab', type=int)
@click.option('--score/--no-score', default=False)
@click.option('--report-input/--no-report-input', default=False)
def nn(emb_path, k, n_vocab, score, report_input):
    logger.info("Loading emb...")
    with open(emb_path) as f:
        words, vecs = load_emb(f, n_vocab)

    vecs = normalize_emb(vecs)

    word2id = get_word2id(words)

    for line in sys.stdin:
        word = line.strip()

        if word not in word2id:
            logger.info("'{}' not in vocab".format(word))
            continue

        idx = word2id[word]
        vec = vecs[idx]

        cosine = vecs.dot(vec)
        cosine[idx] = -100

        rankings = np.argsort(cosine)[::-1]

        if score:
            results = ["{}/{:.3f}".format(words[idx], cosine[idx]) for idx in rankings[:k]]
        else:
            results = ["{}".format(words[idx]) for idx in rankings[:k]]

        if report_input:
            print("{}\t{}".format(word, " ".join(results)))
        else:
            print(" ".join(results))

@cli.command()
@click.option('--batchsize', type=int, default=1000)
@click.option('--n-nearest-neighbors', '-k', type=int, default=10)
def inv_nn(batchsize, n_nearest_neighbors):
    logger.info("Loading")
    words, vecs = load_emb(sys.stdin)

    vecs = normalize_emb(vecs)
    vecs = cupy.array(vecs)

    logger.info("Calculating")
    nn = np.empty((len(words), n_nearest_neighbors), dtype=np.int32)
    for start in tqdm(range(0, len(words), batchsize)):
        end = min(start + batchsize, len(words))

        # (BATCHSIZE, DIM)
        batch_vecs = vecs[start:end]

        batch_words = words[start:end]

        # (BATCHSIZE, VOCABSIZE)
        batch_sim = batch_vecs.dot(vecs.T)

        # (BATCHSIZE, k)
        rank = cupy.argsort(batch_sim, axis=1)[:, -2:-n_nearest_neighbors-2:-1]
        nn[start:end] = cupy.asnumpy(rank)

    inv_nns = defaultdict(list)
    for i in range(len(words)):
        for nn_idx in nn[i]:
            inv_nns[words[nn_idx]].append(words[i])

    for word in words:
        print("{}\t{}".format(word, ' '.join(inv_nns[word])))

@cli.command()
@click.argument('emb-path', type=click.Path(exists=True))
@click.option('--n-vocab', type=int)
def sim(emb_path, n_vocab):
    logger.info("Loading emb...")
    with open(emb_path) as f:
        words, vecs = load_emb(f, n_vocab)
    
    word2id = get_word2id(words)

    for line in sys.stdin:
        word1, word2 = line.strip().split()
        vec1 = vecs[word2id[word1]]
        vec2 = vecs[word2id[word2]]

        cosine = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        print(cosine)

@cli.command()
@click.argument("src-emb-path", type=click.Path(exists=True))
@click.argument("trg-emb-path", type=click.Path(exists=True))
@click.option('--src-n-vocab', type=int)
@click.option('--trg-n-vocab', type=int)
@click.option('--k', type=int, default=10)
@click.option('--score/--no-score', default=False)
@click.option('--report-input/--no-report-input', default=False)
def translate(src_emb_path, trg_emb_path, src_n_vocab, trg_n_vocab,
              k, score, report_input):
    logger.info("Loading src emb...")
    with open(src_emb_path) as f:
        src_words, src_vecs = load_emb(f, src_n_vocab)
        src_vecs = normalize_emb(src_vecs)

    logger.info("Loading trg emb...")
    with open(trg_emb_path) as f:
        trg_words, trg_vecs = load_emb(f, trg_n_vocab)
        trg_vecs = normalize_emb(trg_vecs)

    src_word2idx = get_word2id(src_words)

    for line in sys.stdin:
        src_word = line.strip()
        src_idx = src_word2idx[src_word]
        src_vec = src_vecs[src_idx]

        cosine = trg_vecs.dot(src_vec)

        rankings = np.argsort(cosine)[::-1]

        if score:
            results = ["{}/{:.3f}".format(trg_words[idx], cosine[idx]) for idx in rankings[:k]]
        else:
            results = ["{}".format(trg_words[idx]) for idx in rankings[:k]]

        if report_input:
            print("{}\t{}".format(word, " ".join(results)))
        else:
            print(" ".join(results))





    
@cli.command()
def format():
    words = []
    vecs = []

    dim = None

    for line in sys.stdin:
        word, vec_str = line.strip().split(' ', 1)
        words.append(word)
        vec = np.fromstring(vec_str, sep=' ')

        if dim is not None and dim != len(vec):
            raise Exception("Dimension mismatch")
        else:
            dim = len(vec)

        vecs.append(vec)

    save_emb(sys.stdout, words, vecs)

@cli.command()
@click.option('--n', type=int, default=10)
def head(n):
    _, dim = sys.stdin.readline().split()

    print("{} {}".format(n, dim))

    for i, line in enumerate(sys.stdin):
        if i > n:
            break

        print(line, end='')

if __name__ == '__main__':
    cli()

    
