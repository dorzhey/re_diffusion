import pandas as pd
import muon as mu
import scanpy as sc
from muon import atac as ac
import numpy as np
import warnings
import pychromvar as pc
import sys
import os

from multiprocessing import Pool
import pickle

file_dir = os.path.dirname(__file__)
data_path = os.path.join(file_dir, '..', '..', 're_design', '10x_data')

h5_file_path = os.path.join(data_path, 'pbmc3k_multi.h5mu')
save_path = os.path.join(file_dir, 'generated_data')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mdata = mu.read_h5mu(h5_file_path)

mdata['cre'].var['peak_seq'] = mdata.mod['cre'].uns['peak_seq']
motif_df = mdata['cre'].var.reset_index(names='interval')
motif_df = motif_df[['interval','chrom','peak_seq']]
motif_df.reset_index(inplace=True)
peak_motifs = {}


motif_temp_data_path = os.path.join(file_dir,"motif_temp_data")



def motifs_from_fasta(row):
    name = str(row['index'])
    fasta_path = f"{motif_temp_data_path}/{name}.fasta"
    save_fasta_file = open(fasta_path, "w")
    
    # Sampling sequences
    write_fasta_component = f">{row['interval']}_chrom_{row['chrom']}\n{row['peak_seq']}"
    save_fasta_file.write(write_fasta_component)
    save_fasta_file.close()

    # print("Computing Motifs....")
    bed_path = f"{motif_temp_data_path}/{name}.bed"
    os.system(f"gimme scan {fasta_path} -p JASPAR2020_vertebrates -g hg38 > {bed_path}")
    df_results_seq_guime = pd.read_csv(bed_path, sep="\t", skiprows=5, header=None)
    # extract motif id
    # example: {motif_name "MA0761.2_ETV1" ; motif_instance "AACTCTTCCTGTTT"} ==> {MA0761.2_ETV1}
    df_results_seq_guime["motifs"] = df_results_seq_guime[8].apply(lambda x: x.split('motif_name "')[1].split('"')[0])
    # do not need that, as index is same for each sub dataset and count does not work correct then
    df_results_seq_guime = df_results_seq_guime[[0, "motifs"]].groupby("motifs").count()
    return row['index'], df_results_seq_guime



args_list = [(row,) for _,row in motif_df.iterrows()]


with Pool(4) as pool:
    result_dicts = pool.starmap(motifs_from_fasta, args_list) 
    pool.close()
    pool.join()

for value in result_dicts:
    idx, results = value
    peak_motifs[idx] = results

with open(f"{save_path}/motif_call.pkl", "wb") as file:
    pickle.dump(peak_motifs, file)