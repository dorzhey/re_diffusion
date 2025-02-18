import os
import sys
import numpy as np
from collections import Counter

import blobfile as bf
import torch as th
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW, RMSprop
import wandb

import torchvision.transforms as T

th.manual_seed(3)

file_dir = os.path.dirname(__file__)
data_dir = os.path.join(file_dir, '..','data_preprocessing','generated_data')
save_dir =  os.path.join(file_dir,'classifier_checkpoints')

universal_guide_path = os.path.join(file_dir, '..', '..','re_design', 
                                    'Universal-Guided-Diffusion', 'Guided_Diffusion_Imagenet')
if universal_guide_path not in sys.path:
    sys.path.append(universal_guide_path)


from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.resample import create_named_schedule_sampler

from guided_diffusion.train_util import parse_resume_step_from_filename


classifier_guidance_path =  os.path.join(file_dir, '..', 'classifier_diffusion')
if classifier_guidance_path not in sys.path:
    sys.path.append(classifier_guidance_path)

from guided_tools import (
    set_annealed_lr,
    split_microbatches,
    compute_top_k,
    get_data_generator,
    dict_to_wandb,
)

from universal_tools import DanQ
train_utils_path = os.path.join(file_dir,'..','train_utils')
if train_utils_path not in sys.path:
    sys.path.append(train_utils_path)

from utils_data import load_TF_data, SequenceDataset



class ClassifierTrainingconfig:
    TESTING_MODE = False
    data_dir=f'{data_dir}/tcre_seq_motif_cluster.csv'
    val_data_dir=""
    noised=True
    iterations=5000
    lr=3e-3
    weight_decay=0.05
    anneal_lr=True
    batch_size=1024
    microbatch= -1
    schedule_sampler="uniform"
    resume_checkpoint=""
    log_interval=10
    eval_interval=10
    save_interval=1000

    subset = None
    num_workers = 8
    num_sampling_to_compare_cells = 1000
    
    # Classsifier
    seq_length=200
    image_size = seq_length
    classifier_use_fp16 = False


config = ClassifierTrainingconfig()

dist_util.setup_dist()

if config.TESTING_MODE:
    os.environ['WANDB_SILENT']="true"
    os.environ["WANDB_MODE"] = "offline"
    config.batch_size = 256
    config.microbatch = -1
    
# CLASSIFIER PARAMETERS AND INITIALIZATION 

model = DanQ()
model.cuda()


# DEVICE
model.to(dist_util.dev())


# LOADING SAVED MODEL
resume_step = 0
if config.resume_checkpoint:
    resume_step = parse_resume_step_from_filename(config.resume_checkpoint)
    if dist.get_rank() == 0:
        logger.log(
            f"loading model from checkpoint: {config.resume_checkpoint}... at {resume_step} step"
        )
        model.load_state_dict(
            dist_util.load_state_dict(
                config.resume_checkpoint, map_location=dist_util.dev()
            )
        )

# Needed for creating correct EMAs and fp16 parameters.
dist_util.sync_params(model.parameters())

mp_trainer = MixedPrecisionTrainer(
    model=model, use_fp16=config.classifier_use_fp16, initial_lg_loss_scale=16.0
)

model = DDP(
    model,
    device_ids=[dist_util.dev()],
    output_device=dist_util.dev(),
    broadcast_buffers=False,
    bucket_cap_mb=128,
    find_unused_parameters=False,
)

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
val_data = None

seq_dataset = SequenceDataset(seqs=data["X_train"], c=data["x_train_cell_type"], transform_ddsm = False, transform_dna = None) 
seq_dataset.seqs = seq_dataset.seqs.astype(np.float16 if config.classifier_use_fp16 else np.float32)
seq_dataset.c = seq_dataset.c.long()
train_dataset, valid_dataset = th.utils.data.random_split(seq_dataset, [0.85,0.15])
train_data_loader = get_data_generator(train_dataset, config.batch_size, config.num_workers)
val_data_loader = get_data_generator(valid_dataset, config.batch_size, config.num_workers) #, deterministic=True)
logger.log(f"creating optimizer...")
# opt = AdamW(mp_trainer.master_params, lr=config.lr, weight_decay=config.weight_decay)
cell_num_list = data['cell_types']
cell_list = list(data["numeric_to_tag"].values())

# count cell types in train
cell_dict_temp = Counter(data['x_train_cell_type'].tolist())
# reoder by cell_types list
cell_dict = {k:cell_dict_temp[k] for k in cell_num_list}
# take only counts
cell_type_counts = list(cell_dict.values())
cell_type_probabilities = [x / sum(cell_type_counts) for x in cell_type_counts]
cell_type_weights = th.tensor(cell_type_probabilities, dtype=th.float32, device=dist_util.dev())

opt = AdamW(mp_trainer.master_params, lr=config.lr, weight_decay=config.weight_decay)

if config.resume_checkpoint:
    opt_checkpoint = bf.join(
        bf.dirname(config.resume_checkpoint), f"opt{resume_step:06}.pt"
    )
    logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
    opt.load_state_dict(
        dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
    )

logger.log("training classifier model...")

wandb_dict = {}

loss_fn = nn.CrossEntropyLoss(weight=cell_type_weights, reduction='none')
def forward_backward_log(data_loader, prefix="train"):
    batch, extra = next(data_loader)
    labels = extra.to(dist_util.dev())
    batch = batch.to(dist_util.dev())
    # Noisy images

    for i, (sub_batch, sub_labels) in enumerate(
        split_microbatches(config.microbatch, batch, labels)
    ):
        logits = model(sub_batch)
        #loss = F.cross_entropy(logits, sub_labels, reduction="none")
        loss = loss_fn(logits, sub_labels)
        losses = {}
        losses[f"{prefix}_loss"] = loss.detach()
        losses[f"{prefix}_acc"] = compute_top_k(
            logits, sub_labels, k=1, reduction="none"
        )
        log_loss_dict(losses)
        wandb_dict.update(dict_to_wandb(losses))
        del losses
        loss = loss.mean()
        if loss.requires_grad:
            if i == 0:
                mp_trainer.zero_grad()
            mp_trainer.backward(loss * len(sub_batch) / len(batch))

def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(save_dir, f"danq_model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(save_dir, f"opt{step:06d}.pt"))

def log_loss_dict(losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())

wandb.init(project="guide_classifier")
wandb.config = {"learning_rate": config.lr, "num_sampling_to_compare_cells": config.num_sampling_to_compare_cells, "batch_size": config.batch_size}

for step in range(config.iterations - resume_step):
    logger.logkv("step", step + resume_step)
    logger.logkv(
        "samples",
        (step + resume_step + 1) * config.batch_size * dist.get_world_size(),
    )
    if config.anneal_lr:
        current_lr = set_annealed_lr(opt, config.lr, (step + resume_step) / config.iterations)
        wandb_dict['lr'] = current_lr
    forward_backward_log(train_data_loader)
    mp_trainer.optimize(opt)
    if val_data_loader is not None and not step % config.eval_interval:
        with th.no_grad():
            with model.no_sync():
                model.eval()
                forward_backward_log(val_data_loader, prefix="val")
                model.train()
    if not step % config.log_interval:
        wandb.log(
            wandb_dict, step = step)
        logger.dumpkvs()
    if (
        step
        and dist.get_rank() == 0
        and not (step + resume_step) % config.save_interval
        and not config.TESTING_MODE
    ):
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)

if dist.get_rank() == 0:
    logger.log("saving model...")
    save_model(mp_trainer, opt, step + resume_step)
dist.barrier()
