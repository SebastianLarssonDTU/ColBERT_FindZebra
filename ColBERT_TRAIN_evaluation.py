##############################################################################
#				TRAINING SCORE												 #
##############################################################################

#########################################################
#						Imports							#
#########################################################

import numpy as np
import pandas as pd
import sys
from helper_functions_for_evaluation import *
import os
import subprocess

#########################################################
#						Arguments				 		#
#########################################################

CUDA_DEVICES = sys.argv[1] 					# "1,2,3"
model = sys.argv[2] 						# "Model4_1"
checkpoint = sys.argv[3] 					# "/home/s190619/My_ColBERT/ColBERT_Bio_ClinicalBERT/experiments/BioBERT_msmarco-psg/train.py/BioBERT_msmarco.psg.l2/checkpoints/colbert-200000.dnn"
similarity_measure = sys.argv[4]			# cosine or l2
indexing_checkpoint = "colbert-" + sys.argv[5] + ".dnn"

#########################################################
#						Load data				 		#
#########################################################

# MedQA_corpus_split_full = pd.read_csv("/scratch/s190619/Data_etc/MedQA/MedQA_corpus_split_w_info.tsv") 	# Contains information needed when evaluating
# MedQA_corpus_split_retrieval = pd.read_csv("/scratch/s190619/Data_etc/MedQA/MedQA_corpus_split.tsv") 		# Ready to use for retrievals

MedQA_corpus_split_full = pd.read_csv("/scratch/s190619/Data_etc/MedQA/MedQA_corpus_split_w_info.tsv", encoding="utf8", sep="\t") 	# Contains information needed when evaluating
MedQA_corpus_split_retrieval = "/scratch/s190619/Data_etc/MedQA/MedQA_corpus_split.tsv"						# Ready to use for retrievals

FZ_corpus_split_full = pd.read_csv("/scratch/s190619/Data_etc/FindZebra/FZ_corpus_split_w_info.tsv", encoding="utf8", sep="\t") 	# Contains information needed when evaluating
FZ_corpus_split_retrieval = "/scratch/s190619/Data_etc/FindZebra/FZ_corpus_split.tsv"						# Ready to use for retrievals

MedQA = pd.read_csv("/scratch/s190619/Data_etc/MedQA/disorders_table_train.csv")
MedQA_FZ = MedQA.copy() # Since it has been trained on the whole dataset, I test on the whole dataset, no matter if the disease is found in FZ or not.


#########################################################
#						Variables				 		#
#########################################################

NPROC = len(CUDA_DEVICES.split(","))
EXPERIMENT_PATH = "/scratch/s190619/Data_etc/ColBERT/experiments"
batch_size = calc_batch_size(NPROC) 								# Function from helper file
EXPERIMENT_NAME = model + "-psg" 									# I don't know what psg stands for
EXPERIMENT = model + ".psg." + similarity_measure
# Determine if the number of GPUS makes it distributed (NPROC > 1)
if NPROC > 1:
	DISTRIBUTED = f"-m torch.distributed.launch --nproc_per_node={str(NPROC)}"
else:
	DISTRIBUTED = ""

# Indexing
NTHREADS = "6"
INDEX_PATH = "/scratch/s190619/Data_etc/ColBERT/indexes/"

# Retrievals
QUERIES = "/scratch/s190619/Data_etc/MedQA/queries_medQA_train.tsv"

# Multiple choice
MC = True
num_options = 5
MC_save_dir = "/scratch/s190619/Data_etc/ColBERT/MC_eval_data/"


#########################################################
#						Retrieval				 		#
#########################################################

os.chdir("/home/s190619/My_ColBERT/ColBERT")

retrievals_location = []
eval_task = ["MedQA","FZ"]
for task in eval_task:

	INDEX_NAME = task + "_Corpus_" + model + "." + similarity_measure + "." + str(batch_size) + "x200k"
	INDEXING_CHECKPOINT = EXPERIMENT_PATH + "/" + EXPERIMENT_NAME + "/train.py/" + EXPERIMENT +"/checkpoints/" + indexing_checkpoint
	RETRIEVAL_DIR = "/scratch/s190619/Data_etc/ColBERT/retrievals/"+ "TRAIN_"+INDEX_NAME
	retrievals_location.append("/scratch/s190619/Data_etc/ColBERT/retrievals/"+"TRAIN_"+INDEX_NAME+"/ranking.tsv")

	if not os.path.exists("/scratch/s190619/Data_etc/ColBERT/retrievals/"+"TRAIN_"+INDEX_NAME+"/ranking.tsv"):
		print("Retrieving top 1000 from %s on %s..." %(task,model))

		# Create retrieve command
		retrieve = f"CUDA_VISIBLE_DEVICES={CUDA_DEVICES} \
OMP_NUM_THREADS={NTHREADS} \
python {DISTRIBUTED} \
-m colbert.retrieve \
--amp \
--doc_maxlen 180 \
--mask-punctuation \
--bsize 256 \
--queries {QUERIES} \
--nprobe 30 \
--partitions 32768 \
--faiss_depth 1024 \
--index_root {INDEX_PATH} \
--index_name {INDEX_NAME} \
--checkpoint {INDEXING_CHECKPOINT} \
--root {EXPERIMENT_PATH} \
--experiment {EXPERIMENT_NAME} \
--retrieval_save_path {RETRIEVAL_DIR}"
		os.system(f"{retrieve}")
		print("Done retrieving from %s on %s" %(task,model))
	else:
		print("%s already has retrievals from %s corpus. Skipping." %(model,task))


#########################################################
#			Create MS data from retrievals		 		#
#########################################################

eval_task = ["MedQA","FZ"]
MC_from_retrievals_paths = [MC_save_dir + "from_retrievals/" + "TRAIN_" + model + "_" + "MedQA" + ".tsv",
							MC_save_dir + "from_retrievals/" + "TRAIN_" + model + "_" + "FZ" + ".tsv"]
for task in eval_task:
	if task == "MedQA":
		if not os.path.exists(MC_from_retrievals_paths[0]):
			MC_eval_from_retrievals(retrievals_location[0], MedQA_corpus_split_full, MedQA, MC_from_retrievals_paths[0])
		else:
			print("MC eval data already exists for %s on %s. Skipping..." %(model,task))
	elif task == "FZ":
		if not os.path.exists(MC_from_retrievals_paths[1]):
			MC_eval_from_retrievals(retrievals_location[1], FZ_corpus_split_full, MedQA_FZ, MC_from_retrievals_paths[1])
		else:
			print("MC eval data already exists for %s on %s. Skipping..." %(model,task))
#########################################################
#			Evalutate train retrievals with MC_eval		#
#########################################################

os.chdir("/home/s190619/My_ColBERT/ColBERT_MC")
# try:
# 	CUDA_DEVICES = str(get_free_gpus()[0])
# except:
# 	print("\nERROR: No available gpus. Exiting...", file=sys.stderr)

eval_task = ["MedQA","FZ"]
MC_results_save_path = [MC_save_dir + "results/" + "TRAIN_" + model + "_" + "MedQA" + "_results_MR.csv",
						MC_save_dir + "results/" + "TRAIN_" + model + "_" + "FZ" + "_results_MR.csv"]
for task in eval_task:
	if task == "MedQA":
		if os.path.exists(MC_results_save_path[0]):
			print("MC eval results already exists for %s on %s. Skipping..." %(model,task))
			continue
		MC_triples = MC_from_retrievals_paths[0]

	elif task == "FZ":
		if os.path.exists(MC_results_save_path[1]):
			print("MC eval results already exists for %s on %s. Skipping..." %(model,task))
			continue
		MC_triples = MC_from_retrievals_paths[1]


	MC_eval = f"CUDA_VISIBLE_DEVICES={CUDA_DEVICES} \
python \
-m colbert.test_MC \
--amp \
--doc_maxlen 180 \
--mask-punctuation \
--bsize {batch_size} \
--accum 1 \
--triples {MC_triples} \
--root {EXPERIMENT_PATH} \
--experiment {EXPERIMENT_NAME} \
--similarity {similarity_measure} \
--run {EXPERIMENT} \
--checkpoint {INDEXING_CHECKPOINT} \
--multiple_choice \
--num_options {num_options}"
	
	os.system(f"{MC_eval}")
	print("Done ranking MC from %s on %s" %(task,model))


print("\n\n\nDone!")