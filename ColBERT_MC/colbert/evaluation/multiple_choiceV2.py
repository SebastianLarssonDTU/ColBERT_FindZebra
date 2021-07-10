# Needs : 
#	queries dev/test, 
#	top-N passages retrieved with ColBERT and FAISS
#	answers to the queries
#	options to the queries

# I want it to guess the answer by the top K = 1, 3, 5, 10, and N (if N>10)

# Idea: encode the top K documents together and each option seperately. 
#	Let the documents act as a query and the options as passage, and feed it through
#	ColBERT and then through a softmax function to determine the best answer.

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
import os
import random
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from transformers import AdamW
from colbert.utils.runs import Run
from colbert.utils.amp import MixedPrecisionManager

# from colbert.training.lazy_batcher import LazyBatcher
# from colbert.training.eager_batcher import EagerBatcher
from colbert.evaluation.eager_batcher_MC_test import EagerBatcher_MC
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message
from colbert.training.utils import print_progress_MC, manage_checkpoints, chosen_option_test


def multiple_choice_eval(args):
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    if args.distributed:
        torch.cuda.manual_seed_all(12345)

    if args.distributed:
        assert args.bsize % args.nranks == 0, (args.bsize, args.nranks)
        assert args.accumsteps == 1
        args.bsize = args.bsize // args.nranks

        print("Using args.bsize =", args.bsize, "(per process) and args.accumsteps =", args.accumsteps)

    # if args.lazy:
    #     reader = LazyBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)
    # else:
    #     reader = EagerBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)

    # Get the length of the dataset
    max_iterations = pd.read_csv(args.triples, sep="\t",header=None).shape[0]-50

    reader = EagerBatcher_MC(args, (0 if args.rank == -1 else args.rank), args.nranks)

    if args.rank not in [-1, 0]:
        torch.distributed.barrier()

    colbert = ColBERT.from_pretrained('bert-base-uncased',
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation)

    if args.checkpoint is not None:
        assert args.resume_optimizer is False, "TODO: This would mean reload optimizer too."
        print_message(f"#> Starting from checkpoint {args.checkpoint} -- but NOT the optimizer!")

        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        colbert.load_state_dict(checkpoint['model_state_dict'])

    if args.rank == 0:
        torch.distributed.barrier()

    colbert = colbert.to(DEVICE)
    colbert.train()

    if args.distributed:
        colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[args.rank],
                                                            output_device=args.rank,
                                                            find_unused_parameters=True)

#Added by me | Un-tested | Freeze the layers when fine tuning the model | https://huggingface.co/transformers/training.html
    # if args.freeze_layers:
    #     print("\nFreezing layers")
    #     for param in colbert.parameters():
    #         param.requires_grad = False



    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=args.lr, eps=1e-8)
    optimizer.zero_grad()

    amp = MixedPrecisionManager(args.amp)
    criterion = nn.CrossEntropyLoss()
    # labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = 0.0

    start_batch_idx = 0

    if args.resume:
        assert args.checkpoint is not None
        start_batch_idx = checkpoint['batch']

        reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

    k=0 + args.bsize * args.rank
    results = []
    colbert.eval()
    break_ = False

    with torch.no_grad():

	    for batch_idx, BatchSteps in zip(range(start_batch_idx, args.maxsteps), reader):
	        this_batch_loss = 0.0

	        for queries, passages in BatchSteps:
	            with amp.context():
	                # scores = colbert(queries, passages).view(2, -1)#.permute(1, 0)
	                scores = colbert(queries, passages).view(5, -1).permute(1, 0)
	                # loss = criterion(scores, labels[:scores.size(0)])
	                # loss = loss / args.accumsteps
	            # if args.rank < 1:
	            #     print("Predicted options: ", chosen_option(scores))
	                # print_progress_MC(scores)

	            for i in range(args.bsize):
	            	# print("[%d][%d]\tPredicted options: %d" %(args.rank, k+i, int(chosen_option_test(scores)[i])))
	            	results.append([args.rank, k+i, int(chosen_option_test(scores)[i]), scores.cpu().tolist()[i]])
	            # print("[%d][%d]\tPredicted options: %d" %(args.rank, k+1, int(chosen_option_test(scores)[1])))
	            k += args.bsize * args.nranks

	            # amp.backward(loss)
	            if k % 10000 == 0:
	            	print("Progress: [%d] / [%d]" %(k,max_iterations))

	            if k > max_iterations:
	            	break_ = True
	            	break

	        if break_:
	        	break

	    save_path = args.triples.replace(".tsv","_results_V2_")
	    pd.DataFrame(results, columns = ["gpu_rank","index","predicted_option","scores"]).to_csv(save_path + str(args.rank) + ".csv")
	        # amp.step(colbert, optimizer)

	        # if args.rank < 1:
	        #     avg_loss = train_loss / (batch_idx+1)

	        #     num_examples_seen = (batch_idx - start_batch_idx) * args.bsize * args.nranks
	        #     elapsed = float(time.time() - start_time)

	        #     log_to_mlflow = (batch_idx % 20 == 0)
	        #     Run.log_metric('train/avg_loss', avg_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
	        #     Run.log_metric('train/batch_loss', this_batch_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
	        #     Run.log_metric('train/examples', num_examples_seen, step=batch_idx, log_to_mlflow=log_to_mlflow)
	        #     Run.log_metric('train/throughput', num_examples_seen / elapsed, step=batch_idx, log_to_mlflow=log_to_mlflow)

	        #     print_message(batch_idx, avg_loss)
	        #     manage_checkpoints(args, colbert, optimizer, batch_idx+1)