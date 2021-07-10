import os
import ujson

from functools import partial
from colbert.utils.utils import print_message
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples_MC_train

from colbert.utils.runs import Run


class EagerBatcher_MC():
    def __init__(self, args, rank=0, nranks=1):
        self.rank, self.nranks = rank, nranks
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        self.query_tokenizer = QueryTokenizer(args.query_maxlen)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen)
        self.tensorize_triples = partial(tensorize_triples_MC_train, self.query_tokenizer, self.doc_tokenizer)

        self.triples_path = args.triples
        self._reset_triples()

    def _reset_triples(self):
        self.reader = open(self.triples_path, mode='r', encoding="utf-8")
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        # queries, positives, negatives = [], [], []
        # passage, target, opt1, opt2, opt3, opt4 = [], [], [], [], [], []
        query_t, query_1, query_2, query_3, query_4, positive, negative = [], [], [], [], [], [], []

        for line_idx, line in zip(range(self.bsize * self.nranks), self.reader):
            if (self.position + line_idx) % self.nranks != self.rank:
                continue

            q, pos, neg, t, o1, o2, o3, o4 = line.strip().split('\t')

            query_t.append(q + " " + t)
            query_1.append(q + " " + o1)
            query_2.append(q + " " + o2)
            query_3.append(q + " " + o3)
            query_4.append(q + " " + o4)
            positive.append(pos)
            negative.append(neg)

        self.position += line_idx + 1

        if len(query_t) < self.bsize:
            raise StopIteration

        return self.collate(query_t, query_1, query_2, query_3, query_4, positive, negative)
        # batches = []
        # for i in range(len(passage)):
        #     batches.append((passage[i],[target[i], opt1[i], opt2[i], opt3[i], opt4[i]]))
        # return batches

    def collate(self, query_t, query_1, query_2, query_3, query_4, positive, negative):
        assert len(query_t) == len(positive) == len(negative) == self.bsize

        return self.tensorize_triples(query_t, query_1, query_2, query_3, query_4, positive, negative, self.bsize // self.accumsteps)

    def skip_to_batch(self, batch_idx, intended_batch_size):
        self._reset_triples()

        Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')

        _ = [self.reader.readline() for _ in range(batch_idx * intended_batch_size)]

        return None
