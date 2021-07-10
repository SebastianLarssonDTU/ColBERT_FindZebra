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
models = {"Model5_Q" : [EXPERIMENT_PATH + "/Model5_Q-psg/train.py/Model5_Q.psg.cosine/checkpoints/colbert-1333.dnn", data_triples[1], str(1333)],
		  "Model6_Q" : [EXPERIMENT_PATH + "/Model6_Q-psg/train.py/Model6_Q.psg.cosine/checkpoints/colbert-1000.dnn", data_triples[0], str(1000)],
		  "Model7_Q" : [EXPERIMENT_PATH + "/Model7_Q-psg/train.py/Model7_Q.psg.cosine/checkpoints/colbert-1333.dnn", data_triples[1], str(1333)],
		  "Model8_Q" : [EXPERIMENT_PATH + "/Model8_Q-psg/train.py/Model8_Q.psg.cosine/checkpoints/colbert-1000.dnn", data_triples[0], str(1000)]}


NPROC = len(CUDA_DEVICES.split(","))
batch_size = calc_batch_size(NPROC) 								# Function from helper file
# Determine if the number of GPUS makes it distributed (NPROC > 1)
if NPROC > 1:
	DISTRIBUTED = f"-m torch.distributed.launch --nproc_per_node={str(NPROC)}"
else:
	DISTRIBUTED = ""


for model in models.keys():
	print("\nRunning model \t\t%s \nUsing checkpoint \t%s \nTraining with triples \t%s\n" %(model,models[model][0],models[model][1]))


	EXPERIMENT_NAME = model + "-psg" 									# I don't know what psg stands for
	EXPERIMENT = model + ".psg." + similarity_measure



	test = f"python ColBERT_evaluationV4.py {CUDA_DEVICES} \
	{model} {models[model][0]} {EXPERIMENT_PATH} {similarity_measure} {models[model][2]} 1 1 1 1 1 test"
	
	os.system(f"{test}")