import muon as mu
import scanpy as sc
import os

def visualize_umap(mdata, saved=True):
    file_name='umap_plot.png'
    output_dir='./plots'
    
    # compute the neighbors for only rna
    sc.pp.neighbors(mdata['rna'], n_neighbors=10, n_pcs=20)
    # compute the neighbors for only atac
    sc.pp.neighbors(mdata['atac'], use_rep="X_lsi", n_neighbors=10, n_pcs=20)
    mu.pp.neighbors(mdata, key_added='wnn')
    mu.tl.umap(mdata, neighbors_key='wnn', random_state=10)
    # save the umap coordinates
    mdata.obsm["X_wnn_umap"] = mdata.obsm["X_umap"]
    # cluster the cells with leiden
    sc.tl.leiden(mdata, resolution=.3, neighbors_key='wnn', key_added='leiden_wnn')
    file_path = os.path.join(output_dir, file_name)
    # plot the leiden clusters with umap
    if saved:
        mu.pl.umap(mdata, color=['leiden_wnn'], frameon=False, title="UMAP embedding", legend_loc="on data", save=file_path)
    else:
        mu.pl.umap(mdata, color=['leiden_wnn'], frameon=False, title="UMAP embedding", legend_loc="on data")
    return file_path