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

from colbert.modeling.colbert import ColBERT
from colbert.evaluation.eager_batcher_MC_eval import EagerBatcher_MC
from colbert.evaluation.MC_slow import MC_slow_rerank
from colbert.utils.utils import timestamp, print_message
from colbert.parameters import DEVICE
from colbert.utils.amp import MixedPrecisionManager
from colbert.modeling.inference import ModelInference
from colbert.evaluation.loaders import load_colbert

def multiple_choice(args):
	args.colbert, args.checkpoint = load_colbert(args)
	args.inference = ModelInference(args.colbert, amp=args.amp)
	random.seed(12345)
	np.random.seed(12345)
	torch.manual_seed(12345)
	# if args.distributed:
	#     torch.cuda.manual_seed_all(12345)

	# if args.distributed:
	#     assert args.bsize % args.nranks == 0, (args.bsize, args.nranks)
	#     assert args.accumsteps == 1
	#     args.bsize = args.bsize // args.nranks

	#     print("Using args.bsize =", args.bsize, "(per process) and args.accumsteps =", args.accumsteps)

	reader = EagerBatcher_MC(args, (0 if args.rank == -1 else args.rank), args.nranks)

	# if args.rank not in [-1, 0]:
	#     torch.distributed.barrier()

	# colbert = ColBERT.from_pretrained('bert-base-uncased',
	#                                   query_maxlen=args.query_maxlen,
	#                                   doc_maxlen=args.doc_maxlen,
	#                                   dim=args.dim,
	#                                   similarity_metric=args.similarity,
	#                                   mask_punctuation=args.mask_punctuation)

	# if args.checkpoint is not None:
	#     # assert args.resume_optimizer is False, "TODO: This would mean reload optimizer too."
	#     print_message(f"#> Starting from checkpoint {args.checkpoint} -- but NOT the optimizer!")

	#     checkpoint = torch.load(args.checkpoint, map_location='cpu')
	#     colbert.load_state_dict(checkpoint['model_state_dict'])

	# if args.rank == 0:
	#     torch.distributed.barrier()

	# colbert = colbert.to(DEVICE)

	# if args.distributed:
	#     colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[args.rank],
	                                                        # output_device=args.rank,
	                                                        # find_unused_parameters=True)

	amp = MixedPrecisionManager(args.amp)
	# criterion = nn.CrossEntropyLoss()
	# labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

	start_batch_idx = 0
	answer_indices = []

	k=1
	c=1

	targets = []
	all_ranked_scores = []
	all_ranked_passages = []
	print_when = [1,3,5,10,50,100,500,1000]
	agg_scores = []
	agg_scores_sub = []
	stored_scores = []
	agg_stored_scores_sub_sub = []
	agg_stored_scores_sub = []
	agg_stored_scores = []

	for batch_idx, BatchSteps in zip(range(start_batch_idx, args.maxsteps), reader):
			this_batch_loss = 0.0

			for passages, options in BatchSteps:
				with amp.context():
					scores, ranked_passages, unranked_scores = MC_slow_rerank(args, passages, options)
					# scores = colbert(passages, options).view(int(args.num_options), -1).permute(1, 0)
					#opt_indices = torch.argmax(scores,dim=1).cpu().numpy()
					# print("I'm here!")
					# print(scores)
					targets.append(options[0])
					all_ranked_scores.append(scores)
					all_ranked_passages.append(ranked_passages)
					stored_scores.append(unranked_scores)
					# print("\n\n\n\n\nNow I'm here!")
					# print(ranked_passages)
					if c in print_when:
						# print("\nCorrect guesses at %d:" %(k))
						# print(sum([1 if all_ranked_passages[i][0] == targets[i] else 0 for i in range(len(targets))])/k)
						agg_scores_sub.append(round(sum([1 if all_ranked_passages[i][0] == targets[i] else 0 for i in range(len(all_ranked_scores))])/c,3))
						for i in range(len(stored_scores[0])):
							agg_stored_scores_sub_sub.append(sum([s[i] for s in stored_scores])/c)
						agg_stored_scores_sub.append(agg_stored_scores_sub_sub)
						agg_stored_scores_sub_sub = []
					if c==1000:
						c = 0
						print(int(k/1000), ":", agg_scores_sub)
						agg_scores.append(agg_scores_sub)
						agg_scores_sub = []
						targets = []
						all_ranked_scores = []
						all_ranked_passages = []
						agg_stored_scores.append(agg_stored_scores_sub)
						agg_stored_scores_sub = []
						stored_scores = []
					# if k == 21000:
					# 	results = []
					# 	for j in range(len(print_when)):
					# 		results.append(round(sum([agg_scores[i][j] for i in range(len(agg_scores))])/len(agg_scores),3))
					# 	print(results)
					# 	save_path = args.triples.replace(".tsv","_results_2.csv")
					# 	pd.DataFrame(agg_scores).to_csv(save_path)
					# 	assert 1 == 3
					c+=1
					k+=1
					#answer_indices.append(opt_indices)

	
	results = []
	print("\nResults with majority rule")
	for j in range(len(print_when)):
		results.append(round(sum([agg_scores[i][j] for i in range(len(agg_scores))])/len(agg_scores),3))
	print(results)

	results_by_score = []
	results_by_score2 = []
	print("\nResults with highest overall score")
	for i in range(len(agg_stored_scores)):
		results_by_score.append([np.array(agg_stored_scores[i][j]).argmax() for j in range(len(print_when))])
	for j in range(len(print_when)):
		results_by_score2.append(round(sum([1 if results_by_score[i][j] == 0 else 0 for i in range(len(results_by_score))]) /len(results_by_score),3))
	print(results_by_score2)

	save_path = args.triples.replace(".tsv","_results_MR.csv")
	pd.DataFrame(agg_scores).to_csv(save_path)
	save_path2 = args.triples.replace(".tsv","_results_HOS.csv")
	pd.DataFrame(results_by_score).to_csv(save_path2)
	# textfile = open(save_path, "w")
	# for a in answer_indices:
	#     textfile.write(a + "\n")