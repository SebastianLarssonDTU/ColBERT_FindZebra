import torch
import numpy as np
import pandas as pd
import sys
from helper_functions_for_evaluation import *
import os

CUDA_DEVICES = sys.argv[1] 	
similarity_measure = sys.argv[2]			# cosine or l2


EXPERIMENT_PATH = "/scratch/s190619/pretrained_checkpoints"

checkpoints = [EXPERIMENT_PATH + "/Model1_2colbert-225000.dnn", EXPERIMENT_PATH + "/Model4_2colbert-200000.dnn"]
data_triples = [EXPERIMENT_PATH + "/MedQA_textbooks_split_triples_scrambled.tsv",
				EXPERIMENT_PATH + "/FZ_corpus_split_triples_scrambled.tsv"]
models = {"Model5_Q" : [checkpoints[1], data_triples[1]],
		  "Model6_Q" : [checkpoints[1], data_triples[0]],
		  "Model7_Q" : [checkpoints[0], data_triples[1]],
		  "Model8_Q" : [checkpoints[0], data_triples[0]]}


NPROC = len(CUDA_DEVICES.split(","))
batch_size = calc_batch_size(NPROC) 								# Function from helper file
# Determine if the number of GPUS makes it distributed (NPROC > 1)
if NPROC > 1:
	DISTRIBUTED = f"-m torch.distributed.launch --nproc_per_node={str(NPROC)}"
else:
	DISTRIBUTED = ""

os.chdir("ColBERT")

for model in models.keys():
	print("\nRunning model \t\t%s \nUsing checkpoint \t%s \nTraining with triples \t%s\n" %(model,models[model][0],models[model][1]))


	EXPERIMENT_NAME = model + "-psg" 									# I don't know what psg stands for
	EXPERIMENT = model + ".psg." + similarity_measure



	train = f"CUDA_VISIBLE_DEVICES={CUDA_DEVICES} \
python {DISTRIBUTED} \
-m colbert.train \
--root {EXPERIMENT_PATH} \
--amp \
--accum 1 \
--similarity {similarity_measure} \
--query_maxlen 250 \
--doc_maxlen 75 \
--mask-punctuation \
--run {EXPERIMENT} \
--bsize {batch_size} \
--checkpoint {models[model][0]} \
--triples {models[model][1]} \
--experiment {EXPERIMENT_NAME}"
	
	os.system(f"{train}")