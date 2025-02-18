import os
import sys
import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch 
import pandas as pd
import wandb

file_dir = os.path.dirname(__file__)
data_dir = os.path.join(file_dir, '..','data_preprocessing','generated_data')
save_dir =  os.path.join(file_dir,'train_output')

guided_diff_path = os.path.join(file_dir, '..', '..','re_design', 'guided-diffusion')
if guided_diff_path not in sys.path:
    sys.path.append(guided_diff_path)

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler

train_utils_path = os.path.join(file_dir,'..','train_utils')
if train_utils_path not in sys.path:
    sys.path.append(train_utils_path)
from utils_data import load_TF_data, SequenceDataset
from utils import (
    extract_motifs,
    kl_heatmap,
    generate_heatmap,
    plot_training_loss,
    plot_training_validation    
)


from guided_tools import (
    get_data_generator,
    TrainLoop,
    create_model,
    create_gaussian_diffusion,
    create_classifier
)
class DiffusionTrainingconfig:
    TESTING_MODE = False
    data_dir=f'{data_dir}/tcre_seq_motif_cluster.csv'
    classifier_checkpoint_path = f'{file_dir}/classifier_checkpoints/model045000.pt'
    schedule_sampler="uniform"
    lr=1e-4
    weight_decay=0.0
    lr_anneal_steps=0
    batch_size=512
    microbatch=-1  # -1 disables microbatches
    ema_rate="0.9999"  # comma-separated list of EMA values
    log_interval=200
    sample_interval=200
    save_interval=5000
    resume_checkpoint=""
    use_fp16=False
    fp16_scale_growth=1e-3

    subset = None
    num_workers = 8
    kmer_length = 5
    num_classes = 3
    

    # model
    seq_length=200
    image_size=seq_length
    
    num_channels=128
    num_res_blocks=2
    num_heads=4
    num_heads_upsample=-1
    num_head_channels=-1
    attention_resolutions="100,50,25"
    channel_mult=""
    dropout=0.0
    class_cond=True
    use_checkpoint=False
    use_scale_shift_norm=True
    resblock_updown=True
    use_fp16=False
    use_new_attention_order=False

    # Diffusion
    learn_sigma=False
    diffusion_steps=1000
    noise_schedule="linear"
    timestep_respacing="250"
    use_kl=True
    predict_xstart=False
    rescale_timesteps=False
    rescale_learned_sigmas=False

    # Sampling
    use_ddim=False
    clip_denoised=True
    num_sampling_to_compare_cells = 3000
    sample_bs = 100
    use_classifier = True
    classifier_scale=1 # between 1 and 10 trade off between diversity and fidelity
    run_name = "" # for gimme scan erros when doing several runs simultaneously 

    # Classifier
    classifier_use_fp16=False
    classifier_width=256
    classifier_depth=3
    classifier_attention_resolutions="100,50,25"  # 16
    classifier_use_scale_shift_norm=True  # False
    classifier_resblock_updown=True  # False
    classifier_pool="spatial"
    



config = DiffusionTrainingconfig()
dist_util.setup_dist()
logger.configure()

if config.TESTING_MODE:
    os.environ['WANDB_SILENT']="true"
    os.environ["WANDB_MODE"] = "offline"
    config.batch_size = 32
    config.num_sampling_to_compare_cells = 100
    config.sample_bs = 10
    config.log_interval = 5
    config.sample_interval = 5

logger.log("creating data loader...")
data = load_TF_data(
    data_path=config.data_dir,
    seqlen=config.seq_length,
    limit_total_sequences=config.subset,
    num_sampling_to_compare_cells=config.num_sampling_to_compare_cells,
    to_save_file_name="cre_encode_data_motif_cluster",
    saved_file_name="cre_encode_data_motif_cluster.pkl",
    load_saved_data=True,
    start_label_number = 0,
)

logger.log("creating model and diffusion...")
model = create_model(
        config.image_size,
        config.num_channels,
        config.num_res_blocks,
        channel_mult=config.channel_mult,
        learn_sigma=config.learn_sigma,
        class_cond=config.class_cond,
        use_checkpoint=config.use_checkpoint,
        attention_resolutions=config.attention_resolutions,
        num_heads=config.num_heads,
        num_head_channels=config.num_head_channels,
        num_heads_upsample=config.num_heads_upsample,
        use_scale_shift_norm=config.use_scale_shift_norm,
        dropout=config.dropout,
        resblock_updown=config.resblock_updown,
        use_fp16=config.use_fp16,
        use_new_attention_order=config.use_new_attention_order,
    )
diffusion = create_gaussian_diffusion(
        steps=config.diffusion_steps,
        learn_sigma=config.learn_sigma,
        noise_schedule=config.noise_schedule,
        use_kl=config.use_kl,
        predict_xstart=config.predict_xstart,
        rescale_timesteps=config.rescale_timesteps,
        rescale_learned_sigmas=config.rescale_learned_sigmas,
        timestep_respacing=config.timestep_respacing,
    )
model.to(dist_util.dev())

classifier = None
if config.use_classifier:
    logger.log("loading classifier...")
    classifier = create_classifier(
        image_size = config.image_size,
        classifier_use_fp16 = config.classifier_use_fp16,
        classifier_width = config.classifier_width,
        classifier_depth = config.classifier_depth,
        classifier_attention_resolutions = config.classifier_attention_resolutions,
        classifier_use_scale_shift_norm = config.classifier_use_scale_shift_norm,
        classifier_resblock_updown = config.classifier_resblock_updown,
        classifier_pool = config.classifier_pool,
    )
    classifier.load_state_dict(
        dist_util.load_state_dict(config.classifier_checkpoint_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if config.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()


schedule_sampler = create_named_schedule_sampler(config.schedule_sampler, diffusion)

wandb_config = {"learning_rate": config.lr, "num_sampling_to_compare_cells": config.num_sampling_to_compare_cells, "batch_size": config.batch_size}
wandb.init(project="guided_diffusion", config=wandb_config)

logger.log("training...")
trainloop = TrainLoop(
    config = config,
    model=model,
    diffusion=diffusion,
    classifier=classifier,
    data=data,
    batch_size=config.batch_size,
    microbatch=config.microbatch,
    lr=config.lr,
    ema_rate=config.ema_rate,
    log_interval=config.log_interval,
    sample_interval = config.sample_interval,
    save_interval=config.save_interval,
    resume_checkpoint=config.resume_checkpoint,
    use_fp16=config.use_fp16,
    fp16_scale_growth=config.fp16_scale_growth,
    schedule_sampler=schedule_sampler,
    weight_decay=config.weight_decay,
    lr_anneal_steps=config.lr_anneal_steps,
    run_name = config.run_name,
)
trainloop.run_loop()


validations = [trainloop.all_train_js, trainloop.all_test_js, trainloop.all_shuffle_js]
labels = ["train JS divergence", "test JS divergence", "shuffle JS divergence"]
plot_training_validation(validations, labels, config.sample_interval, save_dir)

print("training graphs saved")


cell_num_list = data['cell_types']
cell_list = list(data["numeric_to_tag"].values())
nucleotides = ["A", "C", "G", "T"]
sample_bs = config.sample_bs
num_samples = round(config.num_sampling_to_compare_cells/sample_bs)
generated_celltype_motif = {}
seqlen = config.seq_length
model.eval()
# iterate over cell types
for cell_num in cell_num_list:
    cell_type = data['numeric_to_tag'][cell_num]
    print(f"Generating {config.num_sampling_to_compare_cells} samples for cell_type {cell_type}")
    final_sequences = []
    for n_a in range(num_samples):
        model_kwargs = {}
        sampled_cell_types = np.array([cell_num] * sample_bs)
        classes = torch.from_numpy(sampled_cell_types).to(dist_util.dev())
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not config.use_ddim else diffusion.ddim_sample_loop
        )
        sampled_images = sample_fn(
            trainloop.model_fn if config.use_classifier else model,
            (config.sample_bs, 1, 4, config.image_size),
            clip_denoised=config.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn = trainloop.cond_fn if config.use_classifier else None,
            device=dist_util.dev()
            ).squeeze(1)
        for n_b, x in enumerate(sampled_images):
            sequence = "".join([nucleotides[s] for s in np.argmax(x.detach().cpu(), axis=0)])
            seq_final = f">seq_test_{n_a}_{n_b}\n" + sequence
            final_sequences.append(seq_final)
        # extract motifs from generated sequences
    df_motifs_count_syn = extract_motifs(final_sequences)
    generated_celltype_motif[cell_type] = df_motifs_count_syn


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

wandb.log({"Generated_Train_js_heatmap":wandb.Image(f"{train_utils_path}/train_data/Generated_Train_js_heatmap.png")})