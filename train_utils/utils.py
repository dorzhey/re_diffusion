import os 
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
import editdistance
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.decomposition import PCA
import re

file_dir = os.path.dirname(__file__)


training_data_path =  os.path.join(file_dir, 'train_data')
#raw_data_dir = os.path.join(file_dir, '..', '..','re_design', '10x_data')
#genome_file = os.path.join(raw_data_dir, 'refdata-gex-GRCh38-2020-A','fasta','genome.fa')
# dna_diff_path = os.path.join(file_dir, '..', '..','re_design', 'DNA-Diffusion', 'src')
# if dna_diff_path not in sys.path:
#     sys.path.append(dna_diff_path)

from utils_data import one_hot_encode, one_hot_encode_dna_sequence

KMER_LENGTH = 5

def extract_text_between_quotes(text):
    if type(text) != str:
        text = str(text)
        print("-"*24, "\n error:")
        print(text)
    match = re.search(r'"(.*?)"', text)
    if match:
        return match.group(1)
    return None

def read_csv_with_skiprows(file_path):
    try:
        df = pd.read_csv(file_path, sep="\t", skiprows=5, header=None)
        # Check if the dataframe is empty after skipping rows
        if df.empty:
            print("The resulting DataFrame is empty.")
            return pd.DataFrame()  # Return an empty DataFrame
        return df
    except pd.errors.EmptyDataError:
        print("EmptyDataError: No data found after skipping rows.")
        return pd.DataFrame()  # Return an empty DataFrame
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame


def calculate_validation_metrics(data, train_sequences, generated_motif, generated_sequences, get_kmer_metrics=True, kmer_length=5):
    train_js = compare_motif_list(generated_motif, data["train_motifs"])
    test_js = compare_motif_list(generated_motif, data["test_motifs"])
    shuffle_js = compare_motif_list(generated_motif, data["shuffle_motifs"])
    gc_ratio = gc_content_ratio(generated_sequences, train_sequences)
    min_edit_distance = min_edit_distance_between_sets(generated_sequences, train_sequences)
    if not get_kmer_metrics:
        return train_js, test_js, shuffle_js, gc_ratio, min_edit_distance
    
    train_vectors, generated_vectors = generate_kmer_frequencies(train_sequences, generated_sequences, kmer_length)
    knn_distance = knn_distance_between_sets(generated_vectors, train_vectors)
    distance_from_closest = distance_from_closest_in_second_set(generated_vectors, train_vectors)

    return train_js, test_js, shuffle_js, gc_ratio, min_edit_distance, knn_distance, distance_from_closest


def extract_motifs(sequence_list: list, run_name=''):
    """Extract motifs from a list of sequences"""
    motifs = open(f"{training_data_path}/{run_name}synthetic_motifs.fasta", "w")
    motifs.write("\n".join(sequence_list))
    motifs.close()
    os.system(f"gimme scan {training_data_path}/{run_name}synthetic_motifs.fasta -p JASPAR2020_vertebrates -g hg38 -n 20 > {training_data_path}/{run_name}syn_results_motifs.bed")    
    df_results_syn = read_csv_with_skiprows(f"{training_data_path}/{run_name}syn_results_motifs.bed")
    if df_results_syn.empty:
        return df_results_syn
    # extract motif id
    # example: {motif_name "MA0761.2_ETV1" ; motif_instance "AACTCTTCCTGTTT"} ==> {MA0761.2_ETV1}
    try:
        df_results_syn["motifs"] = df_results_syn[8].apply(extract_text_between_quotes)
    except Exception as e:
        print(f"An error occurred: {e}")
        print(df_results_syn)
        raise e
    # extract sequence id from sequence list by taking all before last "_"
    # example seq_test_0_0 ==> seq_test_0, which does not nothing for the count
    #df_results_syn[0] = df_results_syn[0].apply(lambda x: x.rpartition("_")[0])
    df_motifs_count_syn = df_results_syn[df_results_syn['motifs'].notna()]
    df_motifs_count_syn = df_motifs_count_syn[[0, "motifs"]].groupby("motifs").count()
    return df_motifs_count_syn

def compare_motif_list(df_motifs_a: pd.DataFrame, df_motifs_b: pd.DataFrame):
    if df_motifs_a.empty or df_motifs_b.empty:
        print("One or both DataFrames are empty. Returning JS divergence of 1.")
        return 1.0
    # Using JS divergence to compare motifs lists distribution
    set_all_mot = set(df_motifs_a.index.values.tolist() + df_motifs_b.index.values.tolist())
    create_new_matrix = []
    for x in set_all_mot:
        list_in = []
        list_in.append(x)  # adding the name
        if x in df_motifs_a.index:
            list_in.append(df_motifs_a.loc[x, 0])
        else:
            list_in.append(1)

        if x in df_motifs_b.index:
            list_in.append(df_motifs_b.loc[x, 0])
        else:
            list_in.append(1)

        create_new_matrix.append(list_in)

    df_motifs = pd.DataFrame(create_new_matrix, columns=["motif", "count_a", "count_b"])

    df_motifs["prob_a"] = df_motifs["count_a"] / (df_motifs["count_a"].sum())
    df_motifs["prob_b"] = df_motifs["count_b"] / (df_motifs["count_b"].sum())
    js_pq = jensenshannon(df_motifs["prob_a"].values, df_motifs["prob_b"].values)
    return np.sum(js_pq)

# too lazy to replace all references of this function to js_heatmap
def kl_heatmap(
    cell_dict1:dict,
    cell_dict2:dict,
    cell_num_list,   
    ):
    final_comp_js = []
    for cell_num1 in cell_num_list:
        comparison_array = []
        motifs1 = cell_dict1[cell_num1]
        for cell_num2 in cell_num_list:
            motifs2 = cell_dict2[cell_num2]
            js_out = compare_motif_list(motifs1, motifs2)
            comparison_array.append(js_out)
        final_comp_js.append(comparison_array)
    return final_comp_js



def generate_heatmap(df_heat: pd.DataFrame, x_label: str, y_label: str, cell_list: list):
    plt.clf()
    plt.rcdefaults()
    plt.rcParams["figure.figsize"] = (15, 15)
    df_plot = pd.DataFrame(df_heat)
    df_plot.columns = cell_list
    df_plot.index = df_plot.columns
    sns.heatmap(df_plot, cmap="Blues_r", annot=True, lw=0.1, vmax=1, vmin=0)
    plt.title(f"JS divergence \n {x_label} sequences x  {y_label} sequences \n MOTIFS probabilities")
    plt.xlabel(f"{x_label} Sequences  \n(motifs dist)")
    plt.ylabel(f"{y_label} \n (motifs dist)")
    plt.grid(False)
    plt.savefig(f"{training_data_path}/{x_label}_{y_label}_js_heatmap.png")





def gc(seq):
    return float(seq.count("G") + seq.count("C")) / len(seq)

def gc_content_ratio(set1, set2):
    return np.mean([gc(seq) for seq in set1])/np.mean([gc(seq) for seq in set2])
    

def min_edit_distance_between_sets(set1, set2):
    """
    Calculate the minimum edit distance for each sequence in set1 to any sequence in set2,
    and return the average of these minimum distances.
    """
    # Calculate pairwise edit distances
    distances = np.array([[editdistance.eval(seq1, seq2) for seq2 in set2] for seq1 in set1])
    
    # Find the minimum distance for each sequence in set1
    min_distances = np.min(distances, axis=1)
    
    # Return the average of these minimum distances
    return np.mean(min_distances)

def generate_kmer_frequencies(set1, set2, k=5, normalize=True, use_pca=True, n_components=50, print_variance=True):
    """
    Generate k-mer frequencies or PCA embeddings for two lists of DNA sequences, ensuring both have the same number of features.
    
    Args:
        set1 (list of str): First list of DNA sequences.
        set2 (list of str): Second list of DNA sequences.
        k (int): Length of k-mers to generate.
        normalize (bool): Whether to normalize k-mer counts by sequence length.
        use_pca (bool): Whether to return PCA embeddings instead of raw frequencies.
        n_components (int): Number of PCA components to return if use_pca is True.
    
    Returns:
        tuple: Tuple containing two numpy arrays:
               - Array of k-mer frequency vectors or PCA embeddings for each sequence in set1.
               - Array of k-mer frequency vectors or PCA embeddings for each sequence in set2.
    """
    # Combine both sets to fit the vectorizer
    combined_sequences = set1 + set2
    # Create all possible k-mers from 'A', 'C', 'G', 'T'
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k))
    
    # Fit the vectorizer on the combined set
    vectorizer.fit(combined_sequences)
    
    # Transform each set separately
    X1 = vectorizer.transform(set1).toarray()
    X2 = vectorizer.transform(set2).toarray()
    
    if normalize:
        set1_length = len(set1[0]) - k + 1
        set2_length = len(set2[0]) - k + 1
        X1 = X1 / set1_length
        X2 = X2 / set2_length
    
    if use_pca:
        # Apply PCA transformation
        pca = PCA(n_components=n_components, svd_solver = "arpack")
        # Fit PCA on the combined data to ensure the same transformation is applied to both sets
        pca.fit(np.vstack((X1, X2)))
        if print_variance:
            print("Fraction of total variance explained by PCA components: ", np.sum(pca.explained_variance_ratio_))
        X1 = pca.transform(X1)
        X2 = pca.transform(X2)
    
    return X1, X2

def knn_distance_between_sets(set1_vectors, set2_vectors, k=15):
    """
    Calculate the KNN distance using k-mer frequencies for each sequence in set1 using the sequences in set2,
    averaging the distances to the k-nearest neighbors.
    
    Args:
        set1 (list of str): List of DNA sequences in the first set.
        set2 (list of str): List of DNA sequences in the second set.
        k (int): Number of nearest neighbors.
    
    Returns:
        float: Average KNN distance.
    """
    # Initialize NearestNeighbors with the second set
    nbrs = NearestNeighbors(n_neighbors= k+ 1, algorithm="ball_tree").fit(set2_vectors)
    
    # Find the distances and indices of the k-nearest neighbors in set2 for each element in set1
    distances, _ = nbrs.kneighbors(set1_vectors)
    distances = distances[:, 1:]
    
    # Compute the mean distance to the k-nearest neighbors for each sequence in set1
    mean_knn_distances = np.mean(distances, axis=1)
    
    # Return the overall average KNN distance
    return np.mean(mean_knn_distances)

def distance_from_closest_in_second_set(set1_vectors, set2_vectors):
    """
    Calculates the Euclidean distance of each sequence in set1 to its nearest sequence in set2
    using k-mer frequency vectors.
    
    Args:
        set1 (list of str): List of DNA sequences in the first set.
        set2 (list of str): List of DNA sequences in the second set.
    
    Returns:
        float: Average distance to the closest sequence in the second set.
    """
    
    # Calculate pairwise distances between all sequences in set1 and set2
    distances = pairwise_distances(set1_vectors, set2_vectors, metric="euclidean")
    
    # Find the minimum distance to any sequence in set2 for each sequence in set1
    min_distances = np.min(distances, axis=1)
    
    # Calculate the average of these minimum distances
    return np.mean(min_distances)


def generate_similarity_metric(seq_len):
    """Capture the syn_motifs.fasta and compare with the  dataset motifs"""
    # nucleotides = ["A", "C", "G", "T"]
    seqs_file = open(f"{training_data_path}/synthetic_motifs.fasta").readlines()
    seqs_to_hotencoder = [one_hot_encode_dna_sequence(s.replace("\n", ""), seq_len).T for s in seqs_file if ">" not in s]

    return seqs_to_hotencoder


def get_best_match(db, x_seq):  # transforming in a function
    return (db * x_seq).sum(1).sum(1).max()


def calculate_mean_similarity(database, input_query_seqs, seq_len=200):
    final_base_max_match = np.mean([get_best_match(database, x) for x in input_query_seqs])
    return final_base_max_match / seq_len


def generate_similarity_using_train(X_train_in, seq_len):
    convert_X_train = X_train_in.copy()
    convert_X_train[convert_X_train == -1] = 0
    generated_seqs_to_similarity = generate_similarity_metric(seq_len)
    return calculate_mean_similarity(convert_X_train, generated_seqs_to_similarity, seq_len)

def plot_training_loss(values, save_dir):
    plt.figure()
    plt.plot(values)
    plt.title(f"Training process \n Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(f"{save_dir}/loss_training.png")

def plot_training_validation(values_list, y_labels, per_epoch, save_dir):
    plt.figure()
    for idx, values in enumerate(values_list):
        X = list(range(0,len(values)*per_epoch,per_epoch))
        plt.plot(X, values, label=y_labels[idx])
    
    plt.title(f"Training process \n Validation stats every {per_epoch} epoch")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(f"{save_dir}/validation_training.png")