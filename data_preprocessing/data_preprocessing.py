import pandas as pd
import muon as mu
import scanpy as sc
from muon import atac as ac
import numpy as np
import os
import warnings
import pychromvar as pc


class DataPreprocessor():
    def __init__(self, h5_file_path = None, ann_file_path = None, genome_file_path = None):
        file_dir = os.path.dirname(__file__)
        self.save_path = os.path.join(file_dir,'..', 'generated_data')

        if h5_file_path is None or ann_file_path is None or genome_file_path is None:
            data_path = os.path.join(file_dir, '..', '..', '..', 're_design', '10x_data')
            h5_file_path = os.path.join(data_path, 'pbmc_granulocyte_sorted_3k_filtered_feature_bc_matrix.h5')
            ann_file_path = os.path.join(data_path, "pbmc_granulocyte_sorted_3k_atac_peak_annotation.tsv")
            genome_file_path = os.path.join(data_path, 'refdata-gex-GRCh38-2020-A','fasta', 'genome.fa')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.mdata = mu.read_10x_h5(h5_file_path)

        self.mdata.var_names_make_unique()
        print("> After make_unique, object has", self.mdata.shape[0], "observations", self.mdata.shape[1], "variables")
        
        self.quality_control()
        print("> After quality control, object has", self.mdata.shape[0], "observations", self.mdata.shape[1], "variables")
        
        self.normalize_data()
        print("> After normalization, object has", self.mdata.shape[0], "observations", self.mdata.shape[1], "variables")
        
        self.merge_ann_data(ann_file_path)
        print("> After merging with annotation, object has", self.mdata.shape[0], "observations", self.mdata.shape[1], "variables")
        
        self.add_peak_sequence(genome_file_path)
        print("> After adding sequences, object has", self.mdata.shape[0], "observations", self.mdata.shape[1], "variables")

        self.cluster_gene = self.create_cluster_gene()
        saved_file_path = self.create_final_dataset()
        print("> Resulting file is save at", saved_file_path)

    def quality_control(self) -> mu.MuData:
        """
        Apply quality control measures to multimodal (RNA and ATAC) data.

        Parameters:
        - mdata (mu.MuData): MuData object containing RNA and ATAC data.

        Returns:
        - mu.MuData: MuData object after applying quality control.
        """
        mdata = self.mdata
        # annotate the group of mitochondrial genes as 'mt'
        mdata['rna'].var['mt'] = mdata['rna'].var_names.str.startswith('MT-')
        # Compute QC metrics for RNA
        sc.pp.calculate_qc_metrics(mdata['rna'], qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

        mu.pp.filter_var(mdata['rna'], 'n_cells_by_counts', lambda x: x >= 3)
        mu.pp.filter_obs(mdata['rna'], 'n_genes_by_counts', lambda x: (x >= 200) & (x < 5000))
        mu.pp.filter_obs(mdata['rna'], 'total_counts', lambda    x: x < 15000)
        mu.pp.filter_obs(mdata['rna'], 'pct_counts_mt', lambda x: x < 20)

        # Compute QC metrics for ATAC
        sc.pp.calculate_qc_metrics(mdata['atac'], percent_top=None, log1p=False, inplace=True)
        # Filter ATAC based on quality metrics
        mu.pp.filter_var(mdata['atac'], 'n_cells_by_counts', lambda x: x >= 50)
        mu.pp.filter_obs(mdata['atac'], 'n_genes_by_counts', lambda x: (x >= 2000) & (x <= 15000))
        mu.pp.filter_obs(mdata['atac'], 'total_counts', lambda x: (x >= 4000) & (x <= 40000))
        # Intersect observations to keep only cells present in both modalities
        mu.pp.intersect_obs(mdata)


    def normalize_data(self) -> mu.MuData:
        """
        Normalize and Scale both scRNA-seq and scATAC-seq data.
        
        Parameters:
        - mdata (mu.MuData): MuData object containing RNA and ATAC data.
        
        Returns:
        - mu.MuData: A MuData object containing normalized and scaled data.
        """
        mdata = self.mdata

        mdata['rna'].layers["counts"] = mdata['rna'].X.copy()
        sc.pp.normalize_total(mdata['rna'], target_sum=1e4)
        sc.pp.log1p(mdata['rna'])

        sc.pp.highly_variable_genes(mdata['rna'], min_mean=0.02, max_mean=4, min_disp=0.5)
        sc.pp.scale(mdata['rna'], max_value=10)
        sc.tl.pca(mdata['rna'], svd_solver='arpack')

        mdata['atac'].layers["counts"] = mdata['atac'].X
        # Transform peak counts with TF-IDF (Term Frequency - Inverse Document Frequency)
        ac.pp.tfidf(mdata['atac'], scale_factor=None)
        # Run Latent Semantic Indexing
        ac.tl.lsi(mdata['atac'])
                
        # # why?
        # mdata['atac'].obsm['X_lsi'] = mdata['atac'].obsm['X_lsi'][:,1:]
        # mdata['atac'].varm["LSI"] = mdata['atac'].varm["LSI"][:,1:]
        # mdata['atac'].uns["lsi"]["stdev"] = mdata['atac'].uns["lsi"]["stdev"][1:]

    def merge_ann_data(self, file_path: str) -> mu.MuData:
        """
        Merge CellRanger peak annotation file to the existing MuData file
        
        Parameters:
        - file_patgh (str): Path to CellRanger peak annotation file
        - mdata (mu.MuData): MuData object containing RNA and ATAC data.
        
        Returns:
        - mu.MuData: A MuData object containing normalized and scaled data.
        """
        mdata = self.mdata
        peak_annotation = pd.read_csv(file_path, sep='\t')
        # Parse peak annotation file and add it to the .uns[“atac”][“peak_annotation”]
        ac.tl.add_peak_annotation(mdata, peak_annotation)

        
    def add_peak_sequence(self, genome_file):
        mdata = self.mdata
        mdata['atac'].X = mdata['atac'].layers["counts"]
        # adds mdata['atac'].uns['peak_seq']
        pc.add_peak_seq(mdata, genome_file=genome_file, delimiter=":|-")
        mdata['atac'].var['peak_seq'] = mdata.mod['atac'].uns['peak_seq']


    def create_cluster_gene(self):
        mdata = self.mdata
        
        # compute the neighbors for only rna
        sc.pp.neighbors(mdata['rna'], n_neighbors=200)
        # compute the neighbors for only atac
        sc.pp.neighbors(mdata['atac'], use_rep="X_lsi", n_neighbors=200)
        mu.pp.neighbors(mdata, key_added='wnn')
        mu.tl.umap(mdata, neighbors_key='wnn', random_state=10)

        # cluster the cells with leiden
        sc.tl.leiden(mdata, resolution=.05, neighbors_key='wnn', key_added='leiden_wnn')
        # plot
        mu.pl.umap(mdata, color=['leiden_wnn'], frameon=False, title="UMAP embedding", legend_loc="on data")

        mdata['rna'].obs['leiden_wnn'] = mdata.obs['leiden_wnn']
        # rank the differentially expressed genes  
        sc.tl.rank_genes_groups(mdata.mod['rna'], 'leiden_wnn', method='wilcoxon')
        # transform into format gene,cluster and take only statistically significant
        from collections import defaultdict 

        filter_by_pvalue = pd.DataFrame(mdata.mod['rna'].uns['rank_genes_groups']['pvals_adj'])<0.05
        de_rna = pd.DataFrame(mdata.mod['rna'].uns['rank_genes_groups']['names'])[filter_by_pvalue]
        pvalues = pd.DataFrame(mdata.mod['rna'].uns['rank_genes_groups']['pvals_adj'])[filter_by_pvalue]
        gene_cluster_pvalues  = defaultdict(dict)
        # flatten into list of unique genes in format gene:cluster 
        for idx, row in de_rna.iterrows():
            for cluster, gene in enumerate(row):
                if gene:
                    gene_cluster_pvalues[gene][int(cluster)] = pvalues.iloc[idx,cluster]

        self.gene_cluster = {}

        for gene, cluter_dict in gene_cluster_pvalues.items():
            # choose cluster with minimum p-value
            self.gene_cluster[gene] = min(cluter_dict, key=cluter_dict.get)
    
    def create_final_dataset(self):
        mdata = self.mdata
        # take annonation data
        ann_data = mdata.mod['atac'].uns['atac']['peak_annotation']
        # get gene as column not index
        ann_data = ann_data.reset_index()
        # merge atac modality with annotation based on peak id in format chr#:start-end
        full_data = mdata['atac'].var.merge( ann_data, left_on='interval', right_on='peak', how='left')
        # subset only required columns
        full_data = full_data[['peak','gene','peak_type','peak_seq']]
        # get chromosome column
        full_data['chrom'] = full_data['peak'].apply(lambda x: x.split(':')[0])
        # new column cluster by mapping genes to cluster with help of gene_cluster dict
        full_data['cell_type'] = full_data['gene'].map(self.gene_cluster)
        # clean
        full_data = full_data[full_data.cell_type.notna()]

        full_data['cell_type'] = "ct"+(full_data.cell_type.astype(int)+1).astype(str)

        # for data constitencty with legacy code
        full_data.rename(columns={'peak_type':'peaktype','peak_seq':'sequence'}, inplace=True)

        file_path = os.path.join(self.save_path, 'new_promoter-distal_seq_rna_class.csv')
        print("File statistics")
        print(full_data['cluster'].value_counts())
        print(full_data['peaktype'].value_counts())
        full_data.to_csv(file_path, index=False)
        return file_path

def preprocess_omics_data(h5_file_path=None, ann_file_path=None, genome_file_path=None):
    processor = DataPreprocessor(h5_file_path, ann_file_path, genome_file_path)
    return processor.mdata
    
if __name__ == "__main__":
    preprocess_omics_data()
