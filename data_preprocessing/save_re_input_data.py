import os
import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
from muon import atac as ac
import pickle

import pychromvar as pc

import pandas as pd

import sys
import os


file_dir = os.path.dirname(__file__)
save_path = os.path.join(file_dir, 'generated_data')
data_path = os.path.join(file_dir, '..', '..', 're_design', '10x_data')
ddsm_path = os.path.join(file_dir, '..', '..', 're_design', 'ddsm')

if ddsm_path not in sys.path:
    sys.path.append(ddsm_path)
    sys.path.append(os.path.join(ddsm_path,"external"))

from ddsm import *
#from selene_utils import *

import datetime
import uuid
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mdata_orig = mu.read_10x_h5(os.path.join(data_path, 'pbmc_granulocyte_sorted_3k_filtered_feature_bc_matrix.h5'))
mdata_orig.var_names_make_unique()
mdata = mdata_orig.copy()

# annotate the group of mitochondrial genes as 'mt'
mdata['rna'].var['mt'] = mdata['rna'].var_names.str.startswith('MT-')

# compute quality
sc.pp.calculate_qc_metrics(mdata['rna'], qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

mu.pp.filter_var(mdata['rna'], 'n_cells_by_counts', lambda x: x >= 3)
mu.pp.filter_obs(mdata['rna'], 'n_genes_by_counts', lambda x: (x >= 200) & (x < 5000))
mu.pp.filter_obs(mdata['rna'], 'total_counts', lambda x: x < 15000)
mu.pp.filter_obs(mdata['rna'], 'pct_counts_mt', lambda x: x < 20)

sc.pp.calculate_qc_metrics(mdata['atac'], percent_top=None, log1p=False, inplace=True)

mu.pp.filter_var(mdata['atac'], 'n_cells_by_counts', lambda x: x >= 50)
mu.pp.filter_obs(mdata['atac'], 'n_genes_by_counts', lambda x: (x >= 2000) & (x <= 15000))
mu.pp.filter_obs(mdata['atac'], 'total_counts', lambda x: (x >= 4000) & (x <= 40000))

# only keep cells that pass the control for both modalities
mu.pp.intersect_obs(mdata)



# normalization

mdata['rna'].layers["counts"] = mdata['rna'].X.copy()
sc.pp.normalize_total(mdata['rna'], target_sum=1e4)
sc.pp.log1p(mdata['rna'])

sc.pp.highly_variable_genes(mdata['rna'], min_mean=0.02, max_mean=4, min_disp=0.5)
sc.pp.scale(mdata['rna'], max_value=10)
sc.tl.pca(mdata['rna'], svd_solver='arpack')

mdata['atac'].layers["counts"] = mdata['atac'].X
ac.pp.tfidf(mdata['atac'], scale_factor=None)
ac.tl.lsi(mdata['atac'])
mdata['atac'].obsm['X_lsi'] = mdata['atac'].obsm['X_lsi'][:,1:]
mdata['atac'].varm["LSI"] = mdata['atac'].varm["LSI"][:,1:]
mdata['atac'].uns["lsi"]["stdev"] = mdata['atac'].uns["lsi"]["stdev"][1:]



# compute the neighbors for only rna
sc.pp.neighbors(mdata['rna'], n_neighbors=10, n_pcs=20)
# compute the neighbors for only atac
sc.pp.neighbors(mdata['atac'], use_rep="X_lsi", n_neighbors=10, n_pcs=20)
# run multimodal nearest neighbors on the graph 
mu.pp.neighbors(mdata, key_added='wnn')

# run multiomdal umap on the wnn
mu.tl.umap(mdata, neighbors_key='wnn', random_state=10)
# save the umap coordinates
mdata.obsm["X_wnn_umap"] = mdata.obsm["X_umap"]
# cluster the cells with leiden
sc.tl.leiden(mdata, resolution=.3, neighbors_key='wnn', key_added='leiden_wnn')
# plot the leiden clusters with umap
# mu.pl.umap(mdata, color=['leiden_wnn'], frameon=False, title="UMAP embedding", legend_loc="on data")



# read cellranger peak annotation
peak_annotation = pd.read_csv(
    os.path.join(data_path, "pbmc_granulocyte_sorted_3k_atac_peak_annotation.tsv"), 
    sep='\t')

# add the interval to match atac var
peak_annotation['interval'] = peak_annotation['chrom'] + ':' + peak_annotation['start'].astype(str) + '-' + peak_annotation['end'].astype(str)

# merge dupes in cellranger peak annotation

def find_match_ind(data_frame, column_name, value_to_match):
    matching_indices = data_frame.index[data_frame[column_name] == value_to_match].tolist()
    return matching_indices

unique_interval = np.unique(peak_annotation['interval'])
ind_col = []
g_col = []
d_col = []
t_col = []
for iu in unique_interval:
    ind = find_match_ind(peak_annotation,'interval',iu)
    ind_col.append(iu)
    g_col.append(peak_annotation.iloc[ind]['gene'].tolist())
    d_col.append(peak_annotation.iloc[ind]['distance'].tolist())
    t_col.append(peak_annotation.iloc[ind]['peak_type'].tolist())

pivot_annotation = {'interval': ind_col, 
                    'gene': g_col, 
                    'distance': d_col, 
                    'peak_type': t_col}
pivot_annotation = pd.DataFrame(pivot_annotation)
    
# merge the cellranger peak annotation to the muon/anndata data
mdata['atac'].var = mdata['atac'].var.merge(pivot_annotation, on='interval', how='left')
# put back the interval var names for the atac modality
mdata.mod['atac'].var_names = mdata.mod['atac'].var.interval




# compute the umap coordinats for only rna
# this adds key mdata.mod['rna'].obsm['X_umap']
sc.tl.umap(mdata.mod['rna'], random_state=10)

sc.tl.leiden(mdata.mod['rna'])
sc.tl.rank_genes_groups(mdata.mod['rna'], 'leiden', method='wilcoxon')
# sc.pl.rank_genes_groups(mdata.mod['rna'], n_genes=25, sharey=False)

# filter the differentially expressed genes  
# rank the differentially expressed genes  
de_rna = pd.DataFrame(mdata.mod['rna'].uns['rank_genes_groups']['names']).iloc[1:1000,:]

file_path = os.path.join(save_path, 'gene_cluster_from_rna_leiden.pkl')

gene_cluster = {}

if os.path.isfile(file_path):
    with open(file_path, 'rb') as file:
        gene_cluster = pickle.load(file)
else:
    with open(file_path, 'wb') as file:
        # Create a dictionary to store the unique values and their corresponding column numbers
        column_numbers = list(range(len(de_rna.columns)))

        for index, row in de_rna.iterrows():
            for col_num, value in enumerate(row):
                if value not in gene_cluster:
                    gene_cluster[value] = col_num
            
        pickle.dump(gene_cluster, file)
        

len(gene_cluster)


# add peak sequences to mdata
################### pip install pychromvar
## import pychromvar as pc 

genome_file = os.path.join(file_dir, '..', '..', 're_design', '10x_data','refdata-gex-GRCh38-2020-A','fasta', 'genome.fa')

mdata['atac'].X = mdata['atac'].layers["counts"]
# adds mdata['atac'].uns['peak_seq']
pc.add_peak_seq(mdata, genome_file=genome_file, delimiter=":|-")
# not used:
pc.add_gc_bias(mdata)
pc.get_bg_peaks(mdata)

    

# iterate over the list of genes/clusters and match the gene name to all entries in the atac.obsm
all_sequences = []
all_clusters = []
all_peaktypes = []
for i,v in gene_cluster.items():
    genebool = np.where([i in a for a in mdata.mod['atac'].var.gene])
    for j in genebool[0]:
        genepos = [k[0] for k in enumerate(mdata.mod['atac'].var.iloc[j,:]['gene']) if k[1]==i]
        peak_type = np.array(mdata.mod['atac'].var.iloc[j,:]['peak_type'])[genepos]
        # sometimes we will have fusion genes with multiple annotations
        promoter_pos = np.where(peak_type == 'promoter')[0]
        if len(promoter_pos)>0:            
            atac_interval = mdata.mod['atac'].var.iloc[j,:]['interval']
            gene_name = np.array(mdata.mod['atac'].var.iloc[j,:]['gene'])[promoter_pos]
            peak_sequence = mdata.mod['atac'].uns['peak_seq'][j]
            if gene_name == i:
                all_sequences.append(peak_sequence)
                all_clusters.append(v)
                all_peaktypes.append('promoter')
        proximal_pos = np.where(peak_type == 'proximal')[0]
        if len(proximal_pos)>0:            
            atac_interval = mdata.mod['atac'].var.iloc[j,:]['interval']
            gene_name = np.array(mdata.mod['atac'].var.iloc[j,:]['gene'])[proximal_pos]
            peak_sequence = mdata.mod['atac'].uns['peak_seq'][j]
            if gene_name == i:
                all_sequences.append(peak_sequence)
                all_clusters.append(v)
                all_peaktypes.append('proximal')
        distal_pos = np.where(peak_type == 'distal')[0]
        if len(distal_pos)>0:            
            atac_interval = mdata.mod['atac'].var.iloc[j,:]['interval']
            gene_name = np.array(mdata.mod['atac'].var.iloc[j,:]['gene'])[distal_pos]
            peak_sequence = mdata.mod['atac'].uns['peak_seq'][j]
            if gene_name == i:
                all_sequences.append(peak_sequence)
                all_clusters.append(v)
                all_peaktypes.append('distal')
            
all_data = {'sequence': all_sequences, 
            'cluster': all_clusters,
            'peaktype': all_peaktypes}
all_data = pd.DataFrame(all_data)
file_path = os.path.join(save_path, 'promoter-distal_seq_rna_class.csv')
all_data.to_csv(file_path)