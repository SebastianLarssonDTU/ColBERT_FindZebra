#########################################################
#						Imports							#
#########################################################

import torch
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
experiment_path = sys.argv[4]
similarity_measure = sys.argv[5]			# cosine or l2
indexing_checkpoint = "colbert-" + sys.argv[6] + ".dnn"
run_indexing = int(sys.argv[7])
run_faiss = int(sys.argv[8])
run_retrieval = int(sys.argv[9])
run_MS_from_retrieval = int(sys.argv[10])
run_ranking = int(sys.argv[11])
train_or_test = (sys.argv[12])

# for i in range(1,13):
# 	print(sys.argv[i])
#########################################################
#						Load data				 		#
#########################################################

# MedQA_corpus_split_full = pd.read_csv("/scratch/s190619/Data_etc/MedQA/MedQA_corpus_split_w_info.tsv") 	# Contains information needed when evaluating
# MedQA_corpus_split_retrieval = pd.read_csv("/scratch/s190619/Data_etc/MedQA/MedQA_corpus_split.tsv") 		# Ready to use for retrievals
if run_indexing == 1:
	MedQA_corpus_split_retrieval = experiment_path + "/MedQA/MedQA_corpus_split.tsv"						# Ready to use for retrievals
	FZ_corpus_split_retrieval = experiment_path + "/FindZebra/FZ_corpus_split.tsv"						# Ready to use for retrievals

if run_MS_from_retrieval == 1:
	MedQA_corpus_split_full = pd.read_csv(experiment_path + "/MedQA/MedQA_corpus_split_w_info.tsv", encoding="utf8", sep="\t") 	# Contains information needed when evaluating
	FZ_corpus_split_full = pd.read_csv(experiment_path + "/FindZebra/FZ_corpus_split_w_info.tsv", encoding="utf8", sep="\t") 	# Contains information needed when evaluating
	
	if train_or_test == "test":
		MedQA = pd.read_csv(experiment_path + "/MedQA/disorders_table_dev-test.csv")
		MedQA_FZ = pd.read_csv(experiment_path + "/MedQA/disorders_table_dev-test_RARE_FZ.csv")

	elif train_or_test == "train":
		MedQA = pd.read_csv(experiment_path + "/MedQA/disorders_table_train.csv")
		MedQA_FZ = pd.read_csv(experiment_path + "/MedQA/disorders_table_train.csv")

#########################################################
#						Variables				 		#
#########################################################

NPROC = len(CUDA_DEVICES.split(","))
EXPERIMENT_PATH = experiment_path
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
INDEX_PATH = experiment_path +"/"

# Retrievals
if train_or_test == "test":
	QUERIES = experiment_path + "/MedQA/queries_medQA_dev-test.tsv"
if train_or_test == "train":
	QUERIES = experiment_path + "/MedQA/queries_medQA_train.tsv"

# Multiple choice
MC = True
num_options = 5
MC_save_dir = experiment_path #"/scratch/s190619/Data_etc/ColBERT/MC_eval_data/"

#Faiss
partitions = 8192

#########################################################
#						Indexing				 		#
#########################################################
os.chdir("/home/s190619/My_ColBERT/ColBERT")

if run_indexing == 1:
	

	
	eval_task = ["MedQA","FZ"]
	for task in eval_task:
		if task == "MedQA":
			COLLECTION = MedQA_corpus_split_retrieval
		elif task == "FZ":
			COLLECTION = FZ_corpus_split_retrieval

		assert os.path.exists(COLLECTION) == True, f"The collection {COLLECTION} doesn't seem to exist"

		INDEX_NAME = task + "_Corpus_" + model + "." + similarity_measure + "." + str(batch_size) + "x200k"
		INDEXING_CHECKPOINT = EXPERIMENT_PATH + "/" + EXPERIMENT_NAME + "/train.py/" + EXPERIMENT +"/checkpoints/" + indexing_checkpoint
		#/home/s190619/My_ColBERT/ColBERT/colbert/index.py
		assert os.path.exists(INDEXING_CHECKPOINT) == True, f"The checkpoint {INDEXING_CHECKPOINT} doesn't seem to exist"

		if not os.path.exists(INDEX_PATH + INDEX_NAME):
			print("Creating index for %s on %s..." %(model,task))

			# Create index
			indexing_command = f"CUDA_VISIBLE_DEVICES={CUDA_DEVICES} \
OMP_NUM_THREADS={NTHREADS} \
python {DISTRIBUTED} \
-m colbert.index \
--root {EXPERIMENT_PATH} \
--amp \
--doc_maxlen 250 \
--doc_maxlen 75 \
--mask-punctuation \
--bsize 256 \
--checkpoint {INDEXING_CHECKPOINT} \
--collection {COLLECTION} \
--index_root {INDEX_PATH} \
--index_name {INDEX_NAME} \
--experiment {EXPERIMENT_NAME}"
			#subprocess.run(f"{indexing_command}")
			os.system(f"{indexing_command}")
			print("Done creating index for %s on %s" %(model,task))
		else:
			print("%s already has index for %s corpus. Skipping." %(model,task))

#########################################################
#						Faiss					 		#
#########################################################

if run_faiss == 1:

	eval_task = ["MedQA","FZ"]
	for task in eval_task:

		INDEX_NAME = task + "_Corpus_" + model + "." + similarity_measure + "." + str(batch_size) + "x200k"
		INDEXING_CHECKPOINT = EXPERIMENT_PATH + "/" + EXPERIMENT_NAME + "/train.py/" + EXPERIMENT +"/checkpoints/" + indexing_checkpoint

		if not os.path.exists(INDEX_PATH + INDEX_NAME + "/ivfpq." + str(partitions) + ".faiss"):
			print("Creating FAISS index for %s on %s..." %(model,task))

			# Create index
			FAISS_command = f"CUDA_VISIBLE_DEVICES={CUDA_DEVICES} \
python {DISTRIBUTED} \
-m colbert.index_faiss \
--index_root {INDEX_PATH} \
--index_name {INDEX_NAME} \
--sample 0.3 \
--partitions {str(partitions)} \
--root {EXPERIMENT_PATH} \
--experiment {EXPERIMENT_NAME}"
			#subprocess.run(f"{indexing_command}")
			os.system(f"{FAISS_command}")
			print("Done creating FAISS index for %s on %s" %(model,task))
		else:
			print("%s already has FAISS index for %s corpus. Skipping." %(model,task))


#########################################################
#						Retrieval				 		#
#########################################################



retrievals_location = []
eval_task = ["MedQA","FZ"]
for task in eval_task:

	INDEX_NAME = task + "_Corpus_" + model + "." + similarity_measure + "." + str(batch_size) + "x200k"
	# INDEXING_CHECKPOINT = EXPERIMENT_PATH + "/" + indexing_checkpoint
	RETRIEVAL_DIR = EXPERIMENT_PATH + "/retrievals/"+INDEX_NAME
	retrievals_location.append(EXPERIMENT_PATH + "/retrievals/"+INDEX_NAME+"/ranking.tsv")

	if run_retrieval == 1:

		if not os.path.exists(EXPERIMENT_PATH + "/retrievals/"+INDEX_NAME+"/ranking.tsv"):
			print("Retrieving top 1000 from %s on %s..." %(task,model))

			# Create retrieve command
			retrieve = f"CUDA_VISIBLE_DEVICES={CUDA_DEVICES} \
OMP_NUM_THREADS={NTHREADS} \
python {DISTRIBUTED} \
-m colbert.retrieve \
--amp \
--query_maxlen 250 \
--doc_maxlen 75 \
--mask-punctuation \
--bsize {str(NPROC)} \
--queries {QUERIES} \
--nprobe 30 \
--partitions {str(partitions)} \
--faiss_depth 256 \
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
MC_from_retrievals_paths = [MC_save_dir + "/" + model + "_" + "MedQA" + ".tsv",
							MC_save_dir + "/" + model + "_" + "FZ" + ".tsv"]

if run_MS_from_retrieval == 1:

	# retrievals_location = []
	# retrievals_location.append("/scratch/s190619/"+ model +"/" + "MedQA" + "_ranking.tsv")
	# retrievals_location.append("/scratch/s190619/"+ model +"/" + "FZ" + "_ranking.tsv")

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
#			Evalutate retrievals with MC		 		#
#########################################################

os.chdir("/home/s190619/My_ColBERT/ColBERT_MC")
# try:
# 	CUDA_DEVICES = str(get_free_gpus()[0])
# except:
# 	print("\nERROR: No available gpus. Exiting...", file=sys.stderr)
MC_from_retrievals_paths = [MC_save_dir + "/" + model + "_" + "MedQA" + ".tsv",
							MC_save_dir + "/" + model + "_" + "FZ" + ".tsv"]
# print(run_ranking)
if run_ranking == 1:

	eval_task = ["MedQA","FZ"]
	MC_results_save_path = [MC_save_dir + "results/" + model + "_" + "MedQA" + "_results_MR.csv",
							MC_save_dir + "results/" + model + "_" + "FZ" + "_results_MR.csv"]
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
python {DISTRIBUTED} \
-m colbert.test_MC_V2 \
--amp \
--query_maxlen 250 \
--doc_maxlen 75 \
--mask-punctuation \
--bsize {str(NPROC*2)} \
--accum 1 \
--triples {MC_triples} \
--root {EXPERIMENT_PATH} \
--experiment {EXPERIMENT_NAME} \
--similarity {similarity_measure} \
--run {EXPERIMENT} \
--checkpoint {checkpoint} \
--multiple_choice \
--num_options {num_options}"
		
		print("\n%s\n" %MC_eval)

		os.system(f"{MC_eval}")
		print("Done ranking MC from %s on %s" %(task,model))

print("\n\n\nDone!")