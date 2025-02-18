import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import sys
import os
import wandb


file_dir = os.path.dirname(__file__)
data_dir = os.path.join(file_dir, '..','data_preprocessing','generated_data')
save_dir = os.path.join(file_dir, 'train_output')
ddsm_path = os.path.join(file_dir, '..', '..', 're_design', 'ddsm')
train_utils_path = os.path.join(file_dir,'..','train_utils')
if ddsm_path not in sys.path:
    sys.path.append(ddsm_path)
    sys.path.append(os.path.join(ddsm_path,"external"))

if train_utils_path not in sys.path:
    sys.path.append(train_utils_path)


from ddsm import (
    GaussianFourierProjection,
    diffusion_factory, 
    diffusion_fast_flatdirichlet,
    gx_to_gv,
    gv_to_gx,
    UnitStickBreakingTransform,
    Euler_Maruyama_sampler
    )
# from selene_utils import *
from utils import (
    extract_motifs,
    compare_motif_list,
    kl_heatmap,
    generate_heatmap,
    generate_similarity_using_train,
    gc_content_ratio,
    min_edit_distance_between_sets,
    generate_kmer_frequencies,
    knn_distance_between_sets,
    distance_from_closest_in_second_set,
    
)

from utils_data import load_TF_data, SequenceDataset

from ddsm_tools import (
    Dense,
    ScoreNet,
    TrainDDSM
)

class ModelParameters:
    diffusion_weights_file = f'{data_dir}/steps400.cat4.time4.0.samples20000.pth'
    datafile = f'{data_dir}/tcre_seq_motif_cluster.csv'
    time_dependent_weights_file = f'{save_dir}/time_dependent_weights.pkl'
    seq_length = 200
    subset = 1000
    device = 'cuda'
    batch_size = 1000
    num_workers = 8
    
    n_time_steps = 400
    speed_balanced = True

    ncat = 4
    num_sampling_to_compare_cells = 100
    sample_bs = 100

config = ModelParameters()


torch.set_default_dtype(torch.float32)



### initialize the dataloader ###
data = load_TF_data(
    data_path=config.datafile,
    seqlen=config.seq_length,
    limit_total_sequences=config.subset,
    num_sampling_to_compare_cells=config.num_sampling_to_compare_cells,
    to_save_file_name="data_subset_forward_process",
    saved_file_name="data_subset_forward_process.pkl",
    load_saved_data=True,
    start_label_number = 0,
)

print("loaded data")
cell_list = list(data["numeric_to_tag"].values())
motif_df = kl_heatmap(
    data['test_motifs_cell_specific'],
    data['train_motifs_cell_specific'],
    cell_list
)
generate_heatmap(motif_df, "Test", "Train", cell_list)
# TRAINING
sampler = Euler_Maruyama_sampler
trainer = TrainDDSM(config, data, sampler)
# return to zeroes from dna diffusion preprocessing
data["X_train"][data["X_train"] == -1] == 0
seq_dataset = SequenceDataset(seqs=data["X_train"], c=data["x_train_cell_type"], transform_dna=None, transform_ddsm = True)
data_loader = DataLoader(seq_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

# weights and presampled noise
### LOAD WEIGHTS
time_dependent_weights = trainer.get_time_dependent_weights(config.time_dependent_weights_file, data_loader, load_saved_file=False)
    

v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints = trainer.v_one, trainer.v_zero, trainer.v_one_loggrad, trainer.v_zero_loggrad, trainer.timepoints

wandb.init(project="ddsm_forward_process")
wandb.config = { "batch_size": config.batch_size,}
wandb.log({"Test_Train_js_heatmap":wandb.Image(f"{train_utils_path}/train_data/Test_Train_js_heatmap.png")})
torch.set_default_dtype(torch.float32)


def diffusion_factory_with_tracking(
        x, time_indices, noise_factory_one, noise_factory_zero,
        noise_factory_one_loggrad, noise_factory_zero_loggrad,
        alpha=None, beta=None, device="cuda", return_v=False, eps=1e-5):
    """
    Generate multivariate Jacobi diffusion samples and scores by sampling from noise factory
    for k-1 Jacobi diffusion processes, recording each step.
    """
    # Initialize alpha and beta if they are not provided
    K = x.shape[-1]
    if alpha is None:
        alpha = torch.ones(K - 1)
    if beta is None:
        beta = torch.arange(K - 1, 0, -1, dtype=torch.float)

    noise_factory_size = noise_factory_one.shape[0]
    sb = UnitStickBreakingTransform()
    sample_indices = torch.randint(0, noise_factory_size, size=x.size()[:-1])
    tracked_samples = []

    for time_idx in time_indices:
        # Expand time index to appropriate shape for broadcasting
        time_ind_expanded = time_idx[(...,) + (None,) * (x.ndim - 2)].expand(x.shape[:-1])

        v_samples = noise_factory_zero[sample_indices, time_ind_expanded, :].to(device).float()
        v_samples_grad = noise_factory_zero_loggrad[sample_indices, time_ind_expanded, :].to(device).float()

        # Iterate over each channel in the last dimension
        for i in range(K - 1):
            inds = x[..., i] == 1
            if torch.any(inds):
                v_samples[..., i][inds] = noise_factory_one[sample_indices[inds], time_ind_expanded[inds], i].to(device).float()
                v_samples_grad[..., i][inds] = noise_factory_one_loggrad[sample_indices[inds], time_ind_expanded[inds], i].to(device).float()
            
        # Convert back from v space to x space, store the samples
        v_samples.requires_grad = True
        samples = sb(v_samples)
        tracked_samples.append(samples.detach().cpu().numpy())  # Store the samples at this time step

    if return_v:
        return v_samples, v_samples_grad, tracked_samples
    else:
        samples_grad = gv_to_gx(v_samples_grad, v_samples)
        samples_grad -= samples_grad.mean(-1, keepdims=True)
        return samples, samples_grad, np.array(tracked_samples)


tracked_samples = torch.FloatTensor(())
every_step = 1
time_indices = torch.arange(start=0,end=config.n_time_steps, step=every_step).reshape((int(config.n_time_steps/every_step),1)) 

print("start_sampling")
for td in tqdm(data_loader):
    samples, grads, tracked_samples_batch = diffusion_factory_with_tracking(
        td[:,:,:4],
        time_indices,
        v_one, v_zero, v_one_loggrad, v_zero_loggrad, trainer.alpha, trainer.beta
    )
    tracked_samples = torch.cat((tracked_samples, torch.FloatTensor(tracked_samples_batch)), dim=1)


nucleotides = ["A", "C", "G", "T"]
KMER_LENGTH = 5

print("generating validation metrics")
for idx, x_train_step in enumerate(tracked_samples):
    decoded_sequences = []
    for n_b, x in enumerate(x_train_step):
            # prepare for fasta and trasform from one-hot to nucletides
            seq_final = f">seq_test_{n_b}\n" + "".join(
                [nucleotides[s] for s in np.argmax(x.detach().cpu(), axis=1)]
            )
            decoded_sequences.append(seq_final)
    # extract motifs from generated sequences
    synt_df = extract_motifs(decoded_sequences)
    seq_similarity = generate_similarity_using_train(data["X_train"], config.seq_length)
    train_kl = compare_motif_list(synt_df, data["train_motifs"])
    test_kl = compare_motif_list(synt_df, data["test_motifs"])
    shuffle_kl = compare_motif_list(synt_df, data["shuffle_motifs"])


    train_sequences = data["train_sequences"]
    gc_ratio = gc_content_ratio(decoded_sequences, train_sequences)
    min_edit_distance = min_edit_distance_between_sets(decoded_sequences, train_sequences)
    train_vectors, generated_vectors = generate_kmer_frequencies(train_sequences, decoded_sequences, KMER_LENGTH, print_variance=False)
    knn_distance = knn_distance_between_sets(generated_vectors, train_vectors)
    distance_from_closest = distance_from_closest_in_second_set(generated_vectors, train_vectors)

    wandb.log(
                {
                    "train_js": train_kl,
                    "test_js": test_kl,
                    "shuffle_js": shuffle_kl,
                    "seq_similarity": seq_similarity,
                    "gc_ratio":gc_ratio,
                    "edit_distance": min_edit_distance,
                    "knn_distance":knn_distance,
                    "distance_endogenous":distance_from_closest
                },
                step=idx+1,
            )