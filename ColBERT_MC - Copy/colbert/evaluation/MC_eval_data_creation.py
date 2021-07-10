import pandas as pd
import ast

queries = "/scratch/s190619/Data_etc/MedQA/disorders_table_dev-test.csv"
retrievals = "/scratch/s190619/Data_etc/ColBERT/retrievals/dev-test/Ranking_Model4_1.tsv"
FZ_passages_path = "/scratch/s190619/Data_etc/FindZebra/collection_FZ_w_titles.tsv"
save_path = "/scratch/s190619/Data_etc/ColBERT/MC_eval_data/" + "Model4_1_dev-test"


q = pd.read_csv(queries)
q["options"] = q["options"].apply(lambda x: ast.literal_eval(x))
opts = []
for i in range(q["options"].shape[0]):
    opt_tmp = [x for x in q["options"].iloc[i].values() if x != q["answer"].iloc[i]]
    opt_tmp.insert(0, q["answer"].iloc[i])
    opts.append(opt_tmp)
q["options"] = opts 
q = q[["qid","options"]]
for i in range(5):
    q[f"options{i}"] = [q.iloc[j]["options"][i] for j in range(q.shape[0])]
q = q.drop("options",axis=1)

r = pd.read_csv(retrievals, 
                sep="\t", header = None, names = ["qid","pid","rank"])

fz = pd.read_csv(FZ_passages_path, encoding="utf8", sep="\t", header=None, names=["pid","title","passage"])

rp = r.join(fz[["pid","passage"]].set_index("pid"), on="pid").join(q.set_index("qid"), on="qid").drop("pid",axis=1)

rp.to_csv(save_path, sep = "\t", index = False, header = False)