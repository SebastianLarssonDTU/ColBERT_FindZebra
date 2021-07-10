import os
import ujson

from functools import partial
from colbert.utils.utils import print_message
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples_MC

from colbert.utils.runs import Run


class EagerBatcher_MC():
    def __init__(self, args, rank=0, nranks=1):
        self.rank, self.nranks = rank, nranks
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        self.query_tokenizer = QueryTokenizer(args.query_maxlen)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen)
        self.tensorize_triples = partial(tensorize_triples_MC, self.query_tokenizer, self.doc_tokenizer)

        self.triples_path = args.triples
        self._reset_triples()

    def _reset_triples(self):
        self.reader = open(self.triples_path, mode='r', encoding="utf-8")
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        # queries, positives, negatives = [], [], []
        passage, target, opt1, opt2, opt3, opt4 = [], [], [], [], [], []

        for line_idx, line in zip(range(self.bsize * self.nranks), self.reader):
            if (self.position + line_idx) % self.nranks != self.rank:
                continue

            _, _, p, t, o1, o2, o3, o4 = line.strip().split('\t')

            passage.append(p)
            target.append(t)
            opt1.append(o1)
            opt2.append(o2)
            opt3.append(o3)
            opt4.append(o4)

        self.position += line_idx + 1

        if len(passage) < self.bsize:
            raise StopIteration

        # return self.collate(passage, target, opt1, opt2, opt3, opt4)
        batches = []
        for i in range(len(passage)):
            batches.append((passage[i],[target[i], opt1[i], opt2[i], opt3[i], opt4[i]]))
        return batches

    def collate(self, passage, target, opt1, opt2, opt3, opt4):
        assert len(passage) == len(target) == len(opt1) == len(opt2) == len(opt3) == len(opt4) == self.bsize

        return self.tensorize_triples(passage, target, opt1, opt2, opt3, opt4, self.bsize // self.accumsteps)

    def skip_to_batch(self, batch_idx, intended_batch_size):
        self._reset_triples()

        Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')

        _ = [self.reader.readline() for _ in range(batch_idx * intended_batch_size)]

        return None
