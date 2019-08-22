import sys

import click
import numpy as np
import cupy
from cupy import cuda
from tqdm import tqdm

from gpu_utils import inv_gpu

LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

logger = logging.getLogger(__name__)

class Mapper:
    def __init__(self, gen_emb, spec_emb, num_neighbors, ignore_exact_words=False):
        self._word2id = {word: i for i, word in enumerate(words)}
        self._gen_emb = gen_emb
        self._gen_emb_norm = gen_emb / cupy.linalg.norm(gen_emb, axis=1)[:, None]
        self._spec_emb = spec_emb
        self._num_neighbors = num_neighbors
        self._ignore_exact_words = ignore_exact_words
        self.output_dim = self._spec_emb.shape[1]

    def get_neighobors(self, batch_emb, batch_words):
        # Normalize batch embeddings
        batch_emb_norm = batch_emb / cupy.linalg.norm(batch_emb, axis=1)[:, None]

        # Compute cosine similarity
        cos_score = batch_emb_norm.dot(self._gen_emb_norm.T)

        # Ignore exact matching words
        if self._ignore_exact_words:
            # indexはbatchの各単語のword indexをもっている
            word_index = cupy.array([self._word2id[word] for word in batch_words if word in self._word2id])
            batch_index = cupy.array([i for i, word in enumerate(batch_words) if word in self._word2id])

            # Set the score of matching words to very small
            cos_score[batch_index, word_index] = -100

        return cupy.argsort(-cos_score, axis=1)[:, :self._num_neighbors]

    def induce_weights(self, batch_emb, nn_idx):
        nn_gen_emb = self._gen_emb[nn_idx]

        diff = batch_emb[:, None] - nn_gen_emb
        C = cupy.einsum('ijk,ilk->ijl', diff, diff)
        C_inv = inv_gpu(C)

        w = cupy.sum(C_inv, axis=1) / cupy.sum(C_inv, axis=(1, 2))[:, None]

        return w

    def __call__(self, batch_emb, batch_words):
        nn_idx = self.get_neighobors(batch_emb, batch_words)

        weights = self.induce_weights(batch_emb, nn_idx)

        nn_spec_emb = self._spec_emb[nn_idx]
        ret = cupy.einsum('ijk,ij->ik', nn_spec_emb, weights)

        return ret

@click.command()
@click.argument('src_emb_path', type=click.Path(exists=True))
@click.argument('trg_emb_path', type=click.Path(exists=True))
@click.option('--num-neighbors', type=int)
@click.option('--batchsize', type=int, default=100)
@click.option('--dictionary-type', type=click.Choice(['exact-match', 'file']))
@click.option('--dictionary', type=click.Path(exists=True))
@click.option('--ignore-exact-words/--no-ignore-exact-words', default=False)
def main(src_emb_path, trg_emb_path, num_neighbors, dictionary_type, dictionary_type, 
        ignore_exact_words=False, batchsize=1000):
    logger.info("Load embeddings....", file=sys.stderr)
    with open(src_emb_path) as f:
        src_emb = utils.load_emb(f)

    with open(trg_emb_path) as f:
        trg_emb = utils.load_emb(f)

    # Extract aligned vectors
    if dictionary_type == 'file':
        src_indicies = []
        trg_indicies = []

        with open(dictionary) as f:
            for line in f:
                src_word, trg_word = line.strip().split()
                src_indicies.append(src_emb.vocab[src_word].index)
                trg_indicies.append(trg_emb.vocab[trg_word].index)

    elif dictionary_type == 'exact-match':
        src_vocab = set(src_emb.vocab.keys())
        trg_vocab = set(trg_emb.vocab.keys())
        shared_vocab = src_vocab.intersection(trg_vocab)

        src_indicies = [src_emb.vocab[word].index for word in shared_vocab]
        trg_indicies = [trg_emb.vocab[word].index for word in shared_vocab]

    src_aligned_vec = src_emb.vectors[src_indicies]
    trg_aligned_vec = trg_emb.vectors[trg_indicies]

    # GPUへ転送
    logger.info("Sending embeddings to GPU...", file=sys.stderr)
    src_aligned_vec = cupy.array(src_aligned_vec)
    trg_aligned_vec = cupy.array(trg_aligned_vec)

    # Mapping Modelを作成
    logger.info("Creating mapping model...", file=sys.stderr)
    mapper = Mapper(src_aligned_vec, trg_aligned_vec, num_neighbors, ignore_exact_words)

    n_vocab, _ = map(int, sys.stdin.readline().split())

    print("{} {}".format(n_vocab, mapper.output_dim))

    for lines, in utils.batchfy((tqdm(enumerate(sys.stdin)),), batchsize=batchsize):
        batch_words, batch_vec_strs = zip(*(line.split(' ', 1) for line in lines))
        batch_vec = cupy.array([np.fromstring(vec_str) for vec_str in batch_vec_strs])

        batch_mapped_vec = mapper(batch_vec, batch_words)

        for i, word in enumerate(batch_words):
            vec_str = ' '.join('{:.6f}'.format(float(v)) for v in batch_mapped_vec[i])
            print('{} {}'.format(word, vec_str))

if __name__ == '__main__':
    main()

