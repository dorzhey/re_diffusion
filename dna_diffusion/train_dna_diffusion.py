import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from accelerate import DataLoaderConfiguration
from accelerate import Accelerator
import wandb

file_dir = os.path.dirname(__file__)
save_dir =  os.path.join(file_dir,'train_output')
data_dir = os.path.join(file_dir, '..','data_preprocessing','generated_data')


dna_diff_path = os.path.join(file_dir, '..', '..','re_design', 'DNA-Diffusion', 'src')
if dna_diff_path not in sys.path:
    sys.path.append(dna_diff_path)

train_utils_path = os.path.join(file_dir,'..','train_utils')
if train_utils_path not in sys.path:
    sys.path.append(train_utils_path)


from dnadiffusion.models.diffusion import Diffusion
from dnadiffusion.models.unet import UNet
from utils import (
    extract_motifs,
    compare_motif_list,
    kl_heatmap,
    generate_heatmap,
    plot_training_loss,
    plot_training_validation
)

from utils_data import load_TF_data

from dna_tools import (
    TrainLoop
)


class ModelParameters:
    TESTING_MODE = False
    datafile = f'{data_dir}/tcre_seq_leiden_cluster_12.csv'
    device = 'cuda'
    batch_size = 720 #960
    num_workers = 8
    seq_length = 200
    subset = None
    num_epochs = 2000
    log_step_show=50
    sample_epoch=50
    save_epoch=100
    num_sampling_to_compare_cells = 3000
    sample_bs = 100
    kmer_length = 5
    run_name = 'leiden12' # for gimme scan erros when doing several runs simultaneously, can leave ''
    
# weights and presampled noise
config = ModelParameters()

if config.TESTING_MODE:
    os.environ['WANDB_SILENT']="true"
    os.environ["WANDB_MODE"] = "offline"
    config.batch_size = 32
    config.num_sampling_to_compare_cells = 100
    config.sample_bs = 10
    config.log_step_show = 1
    config.sample_epoch = 1


data = load_TF_data(
    data_path=config.datafile,
    seqlen=config.seq_length,
    limit_total_sequences=config.subset,
    num_sampling_to_compare_cells=config.num_sampling_to_compare_cells,
    to_save_file_name="cre_encode_data_leiden_cluster_12",
    saved_file_name="cre_encode_data_leiden_cluster_12.pkl",
    load_saved_data=True,
)

cell_list = list(data["numeric_to_tag"].values())

motif_df = kl_heatmap(
    data['train_motifs_cell_specific'],
    data['train_motifs_cell_specific'],
    cell_list
)
generate_heatmap(motif_df, "Train", "Train", cell_list)
motif_df = kl_heatmap(
    data['test_motifs_cell_specific'],
    data['train_motifs_cell_specific'],
    cell_list
)
generate_heatmap(motif_df, "Test", "Train", cell_list)
motif_df = kl_heatmap(
    data['train_motifs_cell_specific'],
    data['shuffle_motifs_cell_specific'],
    cell_list
)
generate_heatmap(motif_df, "Train", "Shuffle", cell_list)
print("loaded data")

wandb.init(project="dnadiffusion")
dataloader_config = DataLoaderConfiguration(split_batches=True)
device_cpu = config.device == "cpu"
accelerator = Accelerator(dataloader_config =dataloader_config, cpu=device_cpu, mixed_precision="fp16", log_with=['wandb'])
# Initialize wandb
accelerator.init_trackers(
    "dnadiffusion_logging",
    config={'batch_size':config.batch_size, "num_sampling_to_compare_cells": config.num_sampling_to_compare_cells},
)

accelerator.log({"Test_Train_js_heatmap":wandb.Image(f"{train_utils_path}/train_data/Test_Train_js_heatmap.png")})

unet = UNet(
        dim=200,
        channels=1,
        dim_mults=(1, 2, 4),
        resnet_block_groups=8,
        num_classes = 15,
    )

diffusion = Diffusion(
    unet,
    timesteps=50,
)

trainloop = TrainLoop(
    config=config,
    data = data,
    model=diffusion,
    accelerator=accelerator,
    epochs=config.num_epochs,
    log_step_show=config.log_step_show,
    sample_epoch=config.sample_epoch,
    save_epoch=config.save_epoch,
    model_name="model_PBMC_3k_3cluster_tcre_leiden_cluster_12",
    num_sampling_to_compare_cells=config.num_sampling_to_compare_cells,
    run_name = config.run_name
)
trainloop.train_loop()
print("training done")

plot_training_loss(trainloop.loss_values, save_dir)
validations = [trainloop.all_train_js, trainloop.all_test_js, trainloop.all_shuffle_js]
labels = ["train JS divergence", "test JS divergence", "shuffle JS divergence"]
plot_training_validation(validations, labels, config.sample_epoch, save_dir)
print("training graphs saved")

# code to only generate heatmaps from saved checkpoint
# trainloop.load("checkpoints/epoch_950_model_PBMC_3k_3cluster_tcre_motif_cluster.pt")
# diffusion = trainloop.accelerator.unwrap_model(trainloop.model)
# diffusion.eval()

# VALIDATION
# iterate over cell types keeping label ratio from train data in generated sequences
_, generated_celltype_motif, _ = trainloop.create_sample_labelwise()

# cell_num_list = data['cell_types']
# nucleotides = ["A", "C", "G", "T"]
# cond_weight_to_metric = 1
# sample_bs = config.sample_bs
# num_samples = round(config.num_sampling_to_compare_cells/sample_bs)
# generated_celltype_motif = {}

# for cell_num in cell_num_list:
#     cell_type = data['numeric_to_tag'][cell_num]
#     final_sequences = []
#     plain_generated_sequences = []
#     cell_type_sample_counts = trainloop.cell_num_sample_counts[cell_num]
#     if cell_type_sample_counts < 100:
#         cell_type_sample_counts = 100
#     cell_type_sample_size = int(cell_type_sample_counts / sample_bs)
#     print(f"Generating {int(cell_type_sample_counts) // sample_bs * sample_bs} samples for cell_type {cell_type}")
#     for n_a in range(cell_type_sample_size):
#         sampled_cell_types = np.array([cell_num] * sample_bs)
#         classes = torch.from_numpy(sampled_cell_types).float().to(diffusion.device)
#         sampled_images = diffusion.sample(classes, (sample_bs, 1, 4, 200), cond_weight_to_metric)
#         for n_b, image in enumerate(sampled_images[-1]):
#             sequence = "".join([nucleotides[s] for s in np.argmax(image.reshape(4, 200), axis=0)])
#             plain_generated_sequences.append(sequence)
#             seq_final = f">seq_test_{cell_num}_{n_a}_{n_b}\n" + sequence
#             final_sequences.append(seq_final)
#     # extract motifs from generated sequences
#     df_motifs_count_syn = extract_motifs(final_sequences, config.run_name)
#     generated_celltype_motif[cell_type] = df_motifs_count_syn

# iterate over cell types in bulk
# for cell_num in cell_num_list:
#     cell_type = data['numeric_to_tag'][cell_num]
#     print(f"Generating {config.num_sampling_to_compare_cells} samples for cell_type {cell_type}")
#     final_sequences = []
#     for n_a in range(num_samples):
#         sampled = torch.from_numpy(np.array([cell_num] * sample_bs))
#         classes = sampled.float().to(diffusion.device)
#         # generate images (time_steps, sample_bs, 1, 4, 200)
#         sampled_images = diffusion.sample(classes, (sample_bs, 1, 4, 200), cond_weight_to_metric)
#         # iterate over last (clear) images
#         for n_b, x in enumerate(sampled_images[-1]):
#             # prepare for fasta and trasform from one-hot to nucletides
#             seq_final = f">seq_test_{n_a}_{n_b}\n" + "".join(
#                 [nucleotides[s] for s in np.argmax(x.reshape(4, 200), axis=0)]
#             )
#             final_sequences.append(seq_final)
#         # extract motifs from generated sequences
#     df_motifs_count_syn = extract_motifs(final_sequences)
#     generated_celltype_motif[cell_type] = df_motifs_count_syn
    



# Generate synthetic vs synthetic heatmap
motif_df = kl_heatmap(
    generated_celltype_motif,
    generated_celltype_motif,
    cell_list
)
generate_heatmap(motif_df, "Generated", "Generated", cell_list)

# Generate synthetic vs train heatmap
motif_df = kl_heatmap(
    generated_celltype_motif,
    data['train_motifs_cell_specific'],
    cell_list
)
generate_heatmap(motif_df, "Generated", "Train", cell_list)

print("Finished generating heatmaps")

accelerator.log({"Generated_Train_js_heatmap":wandb.Image(f"{train_utils_path}/train_data/Generated_Train_js_heatmap.png")})


accelerator.end_training()
