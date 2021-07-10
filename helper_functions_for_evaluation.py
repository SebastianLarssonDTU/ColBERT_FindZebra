import pandas as pd
import ast
import subprocess


def calc_batch_size(NPROC):
	NPROC = int(NPROC)
	batch30 = [1,2,3,5,6]
	batch32 = [4,8]
	batch28 = [7]
	if NPROC in batch30:
		batch_size = 30
	elif NPROC in batch32:
		batch_size = 32
	elif NPROC in batch28:
		print("Warning: 7 processes results in a batch size of 28")
		batch_size = 28
	else: 
		batch_size = 30
		assert 30 % NPROC == 0, "ERROR: Invalid number of processes"
	return batch_size

def clean_string(x):
    return x.replace("\t"," ").replace("\n"," ")

def MC_eval_from_retrievals(retrieval_path, passages, medqa, save_path):
	q = medqa
	q["options"] = q["options"].apply(lambda x: ast.literal_eval(x))
	opts = []
	for i in range(q["options"].shape[0]):
	    opt_tmp = [x for x in q["options"].iloc[i].values() if x != q["answer"].iloc[i]]
	    opt_tmp.insert(0, q["answer"].iloc[i])
	    opts.append(opt_tmp)
	q["options"] = opts 
	q = q[["qid","options"]]
	for i in range(5):
	    q[f"options{i}"] = [clean_string(q.iloc[j]["options"][i]) for j in range(q.shape[0])]
	q = q.drop("options",axis=1)

	r = pd.read_csv(retrieval_path, 
                sep="\t", header = None, names = ["qid","pid","rank"])

	r = r[r["qid"].isin(list(medqa["qid"]))] # Make sure that on the task of FZ, we are only interested in the rare diseases

	passages["passages"] = passages["passages"].apply(lambda x: x.replace("\t"," ").replace("\n", " "))

	#retrieval passages
	rp = r.join(passages[["pid","passages"]].set_index("pid"), on="pid").join(q.set_index("qid"), on="qid").drop("pid",axis=1)
	rp.to_csv(save_path, sep = "\t", index = False, header = False)


def get_free_gpus():
    free_gpus = []
    for line in subprocess.Popen('gpustat',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout:
        if line.decode("utf-8").strip().split("|")[-1] == "":
            gpu = line.decode("utf-8").strip()
            try:
                free_gpus.append(int(gpu[1]))
            except:
                continue
    return free_gpus