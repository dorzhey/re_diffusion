import copy
from typing import Any
import os 
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import pickle
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset



file_dir = os.path.dirname(__file__)

raw_data_dir = os.path.join(file_dir, '..', '..','re_design', '10x_data')
genome_file = os.path.join(raw_data_dir, 'refdata-gex-GRCh38-2020-A','fasta','genome.fa')
training_data_path =  os.path.join(file_dir, 'train_data')
random.seed(10)
np.random.seed(10)
# if dna_diff_path not in sys.path:
#     sys.path.append(dna_diff_path)

def load_TF_data(
    data_path: str,
    seqlen: int = 200,
    saved_file_name: str = "encode_data.pkl",
    limit_total_sequences: int = 0,
    num_sampling_to_compare_cells: int = 1000,
    to_save_file_name = "encode_data",
    load_saved_data: bool = False,
    start_label_number = 1,

):
    if load_saved_data:
        path = f"{training_data_path}/{saved_file_name}"
        with open(path, "rb") as file:
            encode_data = pickle.load(file)

    else:
        encode_data = preprocess_TF_data(
            data_file=data_path,
            seqlen=seqlen,
            subset=limit_total_sequences,
            save_name=to_save_file_name,
            number_of_sequences_to_motif_creation=num_sampling_to_compare_cells
        )
    # Splitting enocde data into train/test/shuffle
    train_motifs = encode_data["train"]["motifs"]
    train_motifs_cell_specific = encode_data["train"]["final_subset_motifs"]

    test_motifs = encode_data["test"]["motifs"]
    test_motifs_cell_specific = encode_data["test"]["final_subset_motifs"]

    shuffle_motifs = encode_data["shuffled"]["motifs"]
    shuffle_motifs_cell_specific = encode_data["shuffled"]["final_subset_motifs"]

    # Creating sequence dataset
    df = encode_data["train"]["df"]
    train_sequences = df['sequence'].values.tolist()
    train_sequences_cell_specific = df.groupby('cell_type')['sequence'].apply(list).to_dict()

    x_train_seq = np.array([one_hot_encode_dna_sequence(x, seqlen) for x in df["sequence"]])
    X_train = np.array([x.T.tolist() for x in x_train_seq])
    X_train[X_train == 0] = -1

    # Creating labels
    numeric_to_tag = dict(enumerate(df["cell_type"].unique(), start_label_number))
    tag_to_numeric = {x: n for n, x in numeric_to_tag.items()}
    
    cell_types = list(numeric_to_tag.keys())
    cell_types.sort()
    x_train_cell_type = torch.tensor([tag_to_numeric[x] for x in df["cell_type"]])

    # Collecting variables into a dict
    encode_data_dict = {
        "train_motifs": train_motifs,
        "train_motifs_cell_specific": train_motifs_cell_specific,
        "test_motifs": test_motifs,
        "test_motifs_cell_specific": test_motifs_cell_specific,
        "shuffle_motifs": shuffle_motifs,
        "shuffle_motifs_cell_specific": shuffle_motifs_cell_specific,
        "tag_to_numeric": tag_to_numeric,
        "numeric_to_tag": numeric_to_tag,
        "cell_types": cell_types,
        "train_sequences": train_sequences,
        "train_sequences_cell_specific":train_sequences_cell_specific,
        "X_train": X_train,
        "x_train_cell_type": x_train_cell_type,
    }

    return encode_data_dict


def preprocess_TF_data(
        data_file,
        seqlen=200,
        subset=None,
        number_of_sequences_to_motif_creation=1000,
        save_name = "encode_data",
        save_output = True
    ):
    data = pd.read_csv(data_file, dtype={'cell_type': str})
    # drop sequences with N
    data = data.drop(data[data.sequence.str.contains("N")].index).reset_index(drop=True)

    # using SCAFE midpoint to cut the sequences:

    if subset is not None:
        # take subset rows of each cell type 
        data = data.sample(subset).reset_index(drop=True)
    # take train and test parts
    # test_subset = [1,3,5,7,9,11,13]
    # test_subset = ['chr'+str(x) for x in test_subset]

    # CUTTING SEQUENCES FOR LAST seqlen BP
    if 'summit_center' in data.columns:
        data = cut_sequences_midpoint(data, seqlen)
    else:
        data['sequence'] = data['sequence'].str[-seqlen:]


    test_data = data[data['chrom'] == "chr1"].reset_index(drop=True)
    # train_data=data.sample(frac=0.5)
    # test_data=data.drop(train_data.index).reset_index(drop=True)
    # train_data=train_data.reset_index(drop=True)

    shuffled_data = data[data['chrom']=='chr2'].reset_index(drop=True)
    shuffled_data["sequence"] = shuffled_data["sequence"].apply(
        lambda x: "".join(random.sample(list(x), len(x)))
    )
    train_data = data[(data["chrom"]!= "chr1") & (data["chrom"] != "chr2")].reset_index(drop=True)
    #train_data = data[~data["chrom"].isin(test_subset)].reset_index(drop=True)
    # Getting motif information from the sequences
    train = generate_motifs_and_fastas(train_data, "train", number_of_sequences_to_motif_creation)
    test = generate_motifs_and_fastas(test_data, "test", number_of_sequences_to_motif_creation)
    shuffled = generate_motifs_and_fastas(shuffled_data,"shuffled",number_of_sequences_to_motif_creation)

    combined_dict = {"train": train, "test": test, "shuffled": shuffled}

    # Writing to pickle
    if save_output:
        # Saving all train, test, shuffled dictionaries to pickle
        with open(f"{training_data_path}/{save_name}.pkl", "wb") as f:
            pickle.dump(combined_dict, f)

    return combined_dict

def cut_sequences_midpoint(data, seqlen):
    half = round(seqlen/2)
    for idx, row in data.iterrows():
        ms = row['summit_center'] - row['start'] # mid - start
        em = row['end'] - row['summit_center'] # end - mid
        seq = row['sequence']
        if ms < half:
            data.loc[idx, 'sequence'] = seq[:seqlen]
        elif em < half:
            data.loc[idx, 'sequence'] = seq[-seqlen:]
        else:
            data.loc[idx, 'sequence'] = seq[ms-half:ms+half]
    return data

def generate_motifs_and_fastas(
    df: pd.DataFrame, 
    name: str, 
    num_sequences: int | None = None
    ) -> dict[str, Any]:
    print("Generating Motifs and Fastas...", name)
    print("---" * 10)

    # Saving fasta
    fasta_path = save_fasta(df, name, num_sequences)

    # Computing motifs
    motifs = motifs_from_fasta(fasta_path)

    # Generating subset specific motifs
    final_subset_motifs = {}
    for comp, v_comp in df.groupby("cell_type"):
        print("Cell_type ", comp)
        v_comp.reset_index(drop=True,inplace=True)
        c_fasta = save_fasta(v_comp, f"{name}_cell_type_{comp}", num_sequences, seq_to_subset_comp=True)
        final_subset_motifs[comp] = motifs_from_fasta(c_fasta)

    return {
        "fasta_path": fasta_path,
        "motifs": motifs,
        "final_subset_motifs": final_subset_motifs,
        "df": df,
    }

def save_fasta(df: pd.DataFrame, name: str, num_sequences: int, seq_to_subset_comp: bool = False) -> str:
    fasta_path = f"{training_data_path}/{name}.fasta"
    save_fasta_file = open(fasta_path, "w")
    
    # Subsetting sequences
    if num_sequences is not None and seq_to_subset_comp and num_sequences < df.shape[0]:
        num_to_sample = num_sequences
    else:
        num_to_sample = df.shape[0]
    #df['idx'] = df.index
    # Sampling sequences
    write_fasta_component = "\n".join(
        df[["peak", "cell_type", "sequence"]]
        .sample(n=num_to_sample)
        .apply(lambda x: f">{x.peak}_cell_type_{x.cell_type}\n{x.sequence}", axis=1)
        .values.tolist()
    )
    save_fasta_file.write(write_fasta_component)
    save_fasta_file.close()

    return fasta_path

def motifs_from_fasta(fasta: str):
    # print("Computing Motifs....")
    os.system(f"gimme scan {fasta} -p JASPAR2020_vertebrates -g hg38 -n 20 > {training_data_path}/train_results_motifs.bed")
    df_results_seq_guime = pd.read_csv(f"{training_data_path}/train_results_motifs.bed", sep="\t", skiprows=5, header=None)
    # extract motif id
    # example: {motif_name "MA0761.2_ETV1" ; motif_instance "AACTCTTCCTGTTT"} ==> {MA0761.2_ETV1}
    df_results_seq_guime["motifs"] = df_results_seq_guime[8].apply(lambda x: x.split('motif_name "')[1].split('"')[0])
    # cut off cell_type to count by peak loci
    df_results_seq_guime[0] = df_results_seq_guime[0].apply(lambda x: x.rpartition("_")[0])
    df_results_seq_guime = df_results_seq_guime[[0, "motifs"]].groupby("motifs").count()
    return df_results_seq_guime




def onehot_to_1channel_image(array):
    return np.expand_dims(array, axis=1)

class SequenceDataset(Dataset):
    def __init__(
        self,
        seqs: np.ndarray,
        c: torch.Tensor,
        random_flip = False,
        transform_dna = onehot_to_1channel_image,
        transform_ddsm = False

    ):
        "Initialization"
        if transform_dna:
            seqs = transform_dna(seqs)
        self.seqs = seqs
        self.c = c
        self.random_flip = random_flip
        
        self.transform_ddsm = transform_ddsm

    def __len__(self):
        "Denotes the total number of samples"
        return self.seqs.shape[0]

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        x = self.seqs[index]
        y = self.c[index]
        if self.random_flip and random.random() < 0.5:
            x = x[:, ::-1]
        

        if self.transform_ddsm:
            return np.concatenate([x.T, np.full((x.shape[1], 1),y)], axis=-1).astype(np.float32)
        
        return x, y

def one_hot_encode_dna_sequence(seq, length):
        # Symmetrically trim the sequence to the specified length
        if len(seq) > length:
            # trim_start = (len(seq) - length) // 2
            # trim_end = trim_start + length
            # seq = seq[trim_start:trim_end]
            seq = seq[:length]
        
        # Define the mapping of nucleotides to one-hot encoding
        nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        # Initialize the one-hot encoded sequence
        one_hot_seq = np.zeros((length, 4), dtype=int)
        
        # Fill in the one-hot encoded sequence
        for i, nucleotide in enumerate(seq):
            if nucleotide in nucleotide_map:
                one_hot_seq[i, nucleotide_map[nucleotide]] = 1
        
        return one_hot_seq

def one_hot_encode(seq, max_seq_len, alphabet):
    """One-hot encode a sequence."""
    alphabet = ["A", "C", "G", "T"]
    seq_len = len(seq)
    seq_array = np.zeros((max_seq_len, len(alphabet)))
    for i in range(seq_len):
        seq_array[i, alphabet.index(seq[i])] = 1
    return seq_array
