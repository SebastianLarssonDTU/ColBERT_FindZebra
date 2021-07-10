import os
import random
import torch
import copy

import colbert.utils.distributed as distributed

from colbert.utils.parser import Arguments
from colbert.utils.runs import Run
from colbert.evaluation.multiple_choiceV2 import multiple_choice_eval


def main():
    parser = Arguments(description='Testing ColBERT with <query target, query options 1-4, retrieved passage> sextuples.')

    parser.add_model_parameters()
    parser.add_model_training_parameters()
    parser.add_training_input()
    parser.add_mc_input_training()

    args = parser.parse()

    assert args.bsize % args.accumsteps == 0, ((args.bsize, args.accumsteps),
                                               "The batch size must be divisible by the number of gradient accumulation steps.")
    assert args.query_maxlen <= 512
    assert args.doc_maxlen <= 512

    args.lazy = args.collection is not None

    with Run.context(consider_failed_if_interrupted=False):
        multiple_choice_eval(args)


if __name__ == "__main__":
    main()

