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
from collections import Counter
from torch import nn
import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch.nn.functional as F
import wandb
from collections import defaultdict
from functools import partial

NUM_CLASS = 3

file_dir = os.path.dirname(__file__)
data_dir = os.path.join(file_dir, '..','data_preprocessing','generated_data')
checkpoints_dir =  os.path.join(file_dir,'diffusion_checkpoints')

universal_guide_path = os.path.join(file_dir, '..', '..','re_design', 
                                    'Universal-Guided-Diffusion', 'Guided_Diffusion_Imagenet')
if universal_guide_path not in sys.path:
    sys.path.append(universal_guide_path)

from guided_diffusion.unet import UNetModel
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.nn import update_ema
from guided_diffusion.resample import LossAwareSampler, UniformSampler
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion import gaussian_diffusion as gd



train_utils_path = os.path.join(file_dir,'..','train_utils')
if train_utils_path not in sys.path:
    sys.path.append(train_utils_path)

from utils import (
    extract_motifs,
    calculate_validation_metrics,
    compare_motif_list,
    kl_heatmap,
    generate_heatmap,
    generate_similarity_using_train,
    gc_content_ratio,
    min_edit_distance_between_sets,
    generate_kmer_frequencies,
    knn_distance_between_sets,
    distance_from_closest_in_second_set,
    plot_training_loss,
    plot_training_validation    
)

from utils_data import load_TF_data, SequenceDataset

def get_data_generator(
    dataset,
    batch_size, 
    num_workers,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
    while True:
        yield from loader

class TrainLoop:
    def __init__(
        self,
        *,
        config,
        operation_config,
        sampling_config,
        model,
        diffusion,
        guide_model,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        sample_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        run_name='',
    ):
        self.model = model
        self.diffusion = diffusion
        self.guide_model = guide_model
        self.config = config
        self.operation_config = operation_config
        self.sampling_config = sampling_config
        
        # data["X_train"][data["X_train"] == -1] = 0
        self.data = data
        self.seq_dataset = SequenceDataset(seqs=data["X_train"], c=data["x_train_cell_type"], transform_ddsm = False)
        self.seq_dataset.seqs = self.seq_dataset.seqs.astype(np.float16 if config.classifier_use_fp16 else np.float32)
        self.seq_dataset.c = self.seq_dataset.c.long()
        self.data_loader = get_data_generator(self.seq_dataset, batch_size, config.num_workers)
        
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.sample_interval = sample_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.current_lr = lr
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.run_name = run_name # for gimme scan erros when doing several runs simultaneously 
        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if torch.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        
        self.cell_num_list = self.data['cell_types']
        cell_list = list(self.data["numeric_to_tag"].values())

        # count cell types in train
        cell_dict_temp = Counter(data['x_train_cell_type'].tolist())
        # reoder by cell_types list
        cell_dict = {k:cell_dict_temp[k] for k in self.cell_num_list}
        # take only counts
        cell_type_counts = list(cell_dict.values())
        self.cell_type_probabilities = [x / sum(cell_type_counts) for x in cell_type_counts]
        cell_num_sample_counts_list = [x * self.config.num_sampling_to_compare_cells for x in self.cell_type_probabilities]
        self.cell_num_sample_counts = {x:y for x,y in zip(self.cell_num_list, cell_num_sample_counts_list)}
        self.data_motif_cell_specific = {cell_type : {"train_motifs":self.data["train_motifs_cell_specific"][cell_type], 
                                                "test_motifs":self.data["test_motifs_cell_specific"][cell_type],
                                                "shuffle_motifs":self.data["shuffle_motifs_cell_specific"][cell_type],
                                                }  for cell_type in cell_list}
        # Metrics
        # self.train_kl, self.test_kl, self.shuffle_kl = 1, 1, 1
        # self.seq_similarity = 1
        # self.min_edit_distance, self.knn_distance, self.distance_from_closest = 100, 0, 0
        # self.gc_ratio = 0.5
        # for plots
        self.loss_values = []
        self.all_train_js, self.all_test_js, self.all_shuffle_js = [],[],[]
        self.all_seq_similarity = []

        self.log_dict = {}

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data_loader)

            self.run_step(batch, cond)

            if self.step> 0 and self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if self.step> 0 and self.step % self.sample_interval == 0:
                logger.log(f"sampling on step {self.step}")
                
                generated_motif, generated_celltype_motif, generated_celltype_sequences = self.create_sample_labelwise()
                generated_sequences = [sequence for sequences in generated_celltype_sequences.values() for sequence in sequences]
                train_sequences = self.data["train_sequences"]
                # first calculate validation metrics in bulk
                seq_similarity = generate_similarity_using_train(self.data["X_train"], self.config.seq_length)
                #train_js, test_js, shuffle_js, gc_ratio, min_edit_distance, knn_distance, distance_from_closest = calculate_validation_metrics(self.data, train_sequences, generated_motif, generated_sequences, get_kmer_metrics = True, kmer_length = self.config.kmer_length,)
                train_js, test_js, shuffle_js, gc_ratio, min_edit_distance = calculate_validation_metrics(
                    self.data, train_sequences, generated_motif, generated_sequences, get_kmer_metrics = False, kmer_length = self.config.kmer_length,
                    )
                self.log_dict.update({
                    "train_js": train_js,
                    "test_js": test_js,
                    "shuffle_js": shuffle_js,
                    "edit_distance": min_edit_distance,
                    # "knn_distance":knn_distance,
                    # "distance_endogenous":distance_from_closest,
                    "seq_similarity": seq_similarity,
                    "gc_ratio":gc_ratio,
                })
                self.all_seq_similarity.append(seq_similarity)
                self.all_train_js.append(train_js)
                self.all_test_js.append(test_js)
                self.all_shuffle_js.append(shuffle_js)
                print("Similarity", seq_similarity)
                print("JS_TRAIN", train_js)
                print("JS_TEST", test_js)
                print("JS_SHUFFLE", shuffle_js)                
                print("GC_ratio", gc_ratio)
                print("Min_Edit", min_edit_distance)
                # print("KNN_distance", knn_distance)
                # print("Distance_endogenous", distance_from_closest)
                # now, generate metrics cell_type-wise and then average it
                validation_metric_cell_specific = defaultdict(list)
                for cell_num in self.cell_num_list:
                    cell_type = self.data['numeric_to_tag'][cell_num]
                    new_data = self.data_motif_cell_specific[cell_type]
                    train_sequences_cell_specific = self.data["train_sequences_cell_specific"][cell_type]
                    generated_motif_cell_specific = generated_celltype_motif[cell_type]
                    generated_sequences_cell_specific = generated_celltype_sequences[cell_type]
                    #train_js, test_js, shuffle_js, gc_ratio, min_edit_distance, knn_distance, distance_from_closest = calculate_validation_metrics(new_data, train_sequences_cell_specific, generated_motif_cell_specific, generated_sequences_cell_specific, self.config.kmer_length)
                    train_js, test_js, shuffle_js, gc_ratio, min_edit_distance = calculate_validation_metrics(
                        new_data, train_sequences_cell_specific, generated_motif_cell_specific, generated_sequences_cell_specific, get_kmer_metrics=False, kmer_length= self.config.kmer_length)
                    # adding to dict
                    validation_metric_cell_specific["train_js_by_cell_type"].append(train_js)
                    validation_metric_cell_specific["test_js_by_cell_type"].append(test_js)
                    validation_metric_cell_specific["shuffle_js_by_cell_type"].append(shuffle_js)
                    validation_metric_cell_specific["min_edit_distance_by_cell_type"].append(min_edit_distance)
                    # validation_metric_cell_specific["knn_distance_by_cell_type"].append(knn_distance)
                    # validation_metric_cell_specific["distance_from_closest_by_cell_type"].append(distance_from_closest)
                    validation_metric_cell_specific["gc_ratio_by_cell_type"].append(gc_ratio)
                    

                validation_metric_average = {x:np.mean(y) for x,y in validation_metric_cell_specific.items()}
                # saving to logging dictionary
                self.log_dict.update(validation_metric_average)
                print("cell type-wise \n",validation_metric_average)                
                

            
                # self.seq_similarity = generate_similarity_using_train(self.data["X_train"], self.config.seq_length)
                # self.train_kl = compare_motif_list(generated_motif, self.data["train_motifs"])
                # self.test_kl = compare_motif_list(generated_motif, self.data["test_motifs"])
                # self.shuffle_kl = compare_motif_list(generated_motif, self.data["shuffle_motifs"])
                # self.gc_ratio = gc_content_ratio(generated_sequences, train_sequences)
                # self.min_edit_distance = min_edit_distance_between_sets(generated_sequences, train_sequences)
                # train_vectors, generated_vectors = generate_kmer_frequencies(train_sequences, generated_sequences, self.config.kmer_length)
                # self.knn_distance = knn_distance_between_sets(generated_vectors, train_vectors)
                # self.distance_from_closest = distance_from_closest_in_second_set(generated_vectors, train_vectors)
                
            if self.step % self.log_interval == 0:
                self.log_dict.update({"loss": self.avg_loss, "lr":self.current_lr})
                wandb.log(self.log_dict, step=self.step)
                logger.dumpkvs()
            
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self.current_lr = self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            cond = {"y":cond}
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            self.avg_loss = loss.item()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
        return lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(checkpoints_dir, filename), "wb") as f:
                    torch.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(checkpoints_dir, f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                torch.save(self.opt.state_dict(), f)

        dist.barrier()

    def create_sample_labelwise(self):
        self.model.eval()
        config = self.config
        nucleotides = ["A", "C", "G", "T"]
        generated_celltype_motif = {}
        generated_celltype_sequences = {}
        bulk_final_sequences = []
        for cell_num in self.cell_num_list:
            cell_type = self.data['numeric_to_tag'][cell_num]
            final_sequences = []
            plain_generated_sequences = []
            cell_type_sample_size = int(self.cell_num_sample_counts[cell_num] / config.sample_bs)
            print(f"Generating {int(self.cell_num_sample_counts[cell_num]) // config.sample_bs * config.sample_bs} samples for cell_type {cell_type}")
            for n_a in range(cell_type_sample_size):
                model_kwargs = {}
                sampled_cell_types = np.array([cell_num] * config.sample_bs)
                classes = torch.from_numpy(sampled_cell_types).to(dist_util.dev())
                model_kwargs["y"] = classes
                sampled_images = self.diffusion.ddim_sample_loop_operation(
                    partial(self.model_fn, model=self.model, args=config),
                    (config.sample_bs, 1, 4, config.image_size),
                    operated_image=None,
                    operation=self.operation_config,
                    clip_denoised=config.clip_denoised,
                    model_kwargs=model_kwargs,
                    cond_fn=partial(self.cond_fn),
                    device=torch.device('cuda'),
                    progress=self.sampling_config.progressive
                ).squeeze(1)
                for n_b, x in enumerate(sampled_images):
                    sequence = "".join([nucleotides[s] for s in np.argmax(x.detach().cpu(), axis=0)])
                    plain_generated_sequences.append(sequence)
                    seq_final = f">seq_test_{n_a}_{n_b}\n" + sequence
                    final_sequences.append(seq_final)
            bulk_final_sequences += final_sequences
            # extract motifs from generated sequences
            df_motifs_count_syn = extract_motifs(final_sequences, self.run_name)
            generated_celltype_motif[cell_type] = df_motifs_count_syn
            generated_celltype_sequences[cell_type] = plain_generated_sequences
        generated_motif = extract_motifs(bulk_final_sequences, self.run_name)
        self.model.train()
        return generated_motif, generated_celltype_motif, generated_celltype_sequences


    def cond_fn(self, x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = self.guide_model(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * self.config.classifier_scale

    def model_fn(self, x, t, y=None):
        assert y is not None
        return self.model(x, t, y if self.config.class_cond else None)


def dict_to_wandb(losses):
    result = {}
    for key, values in losses.items():
        result[key] = values.mean().item()

    return result

def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr

def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0

def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None

def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def create_model_and_diffusion(config):
        config = config
        logger.log("creating model and diffusion...")
        channel_mult = (1, 2, 3)
        attention_ds = []
        for res in config.attention_resolutions.split(","):
            attention_ds.append(config.image_size // int(res))
        NUM_CLASSES = 10
        model = UNetModel(
            image_size=config.image_size,
            in_channels=1,
            model_channels=config.num_channels,
            out_channels=(1 if not config.learn_sigma else 3),
            num_res_blocks=config.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=config.dropout,
            channel_mult=channel_mult,
            dims=2,
            num_classes=(NUM_CLASSES if config.class_cond else None),
            use_checkpoint=config.use_checkpoint,
            use_fp16=config.use_fp16,
            num_heads=config.num_heads,
            num_head_channels=config.num_head_channels,
            num_heads_upsample=config.num_heads_upsample,
            use_scale_shift_norm=config.use_scale_shift_norm,
            resblock_updown=config.resblock_updown,
            use_new_attention_order=config.use_new_attention_order,
        )
        model.to(dist_util.dev())

        betas = gd.get_named_beta_schedule(config.noise_schedule, config.diffusion_steps)
        if config.use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif config.rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE
        if not config.timestep_respacing:
            timestep_respacing = [config.diffusion_steps]
        
        diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(config.diffusion_steps, timestep_respacing),
            betas=betas,
            model_mean_type=(
                gd.ModelMeanType.EPSILON if not config.predict_xstart else gd.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gd.ModelVarType.FIXED_LARGE
                    if not config.sigma_small
                    else gd.ModelVarType.FIXED_SMALL
                )
                if not config.learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            rescale_timesteps=config.rescale_timesteps,
        )

        return model, diffusion
        

        




class OperationArgs:
    def __init__(self):
        self.num_steps = [1]
        self.operation_func = None
        self.optimizer = 'adamw'
        self.lr = 1e-2
        self.loss_func = None
        self.max_iters = 0
        self.loss_cutoff = 0.00001
        self.lr_scheduler = None
        self.warm_start = False
        self.old_img = None
        self.fact = 0.5
        self.print = False
        self.print_every = None
        self.folder = './temp/'
        self.tv_loss = None
        self.guidance_3 = False
        self.optim_guidance_3_wt = 2.0
        self.other_guidance_func = None
        self.other_criterion = None
        self.original_guidance = False
        self.sampling_type = None
        self.loss_save = None

class SamplingArgs:
    def __init__(self):
        self.num_steps = [1]
        self.operation_func = None
        self.optimizer = 'adamw'
        self.lr = 1e-4
        self.loss_func = None
        self.max_iters = 0
        self.loss_cutoff = 0.00001
        self.lr_scheduler = None
        self.warm_start = False
        self.old_img = None
        self.fact = 0.5
        self.print = False
        self.print_every = None
        self.folder = './temp/'
        self.tv_loss = None
        self.guidance_3 = False
        self.optim_guidance_3_wt = 2.0
        self.other_guidance_func = None
        self.other_criterion = None
        self.original_guidance = False
        self.sampling_type = None
        self.loss_save = None
        self.root = '/fs/cml-datasets/ImageNet/ILSVRC2012'
        self.momentum = 0.9
        self.wd = 1e-2
        self.shuffle = False
        self.direct = False
        self.epochs = 100
        self.run_command = 'PYTHONPATH=. python Guided/membership_classification.py'
        self.save_every = 1
        self.test_every = 10
        self.use_noise = False
        self.fixed_noise = False
        self.almost_fixed_noise = False
        self.distribution = False
        self.load = False
        self.remove_bn = False
        self.repeat = 1
        self.use_image = 1
        self.wandb = 1
        self.input_size = 64
        self.optim_lr = 1e-2
        self.optim_backward_guidance_max_iters = 0
        self.optim_loss_cutoff = 0.00001
        self.optim_forward_guidance = False
        self.optim_original_conditioning = False
        self.optim_forward_guidance_wt = 2.0
        self.optim_tv_loss = None
        self.optim_warm_start = False
        self.optim_print = False
        self.optim_folder = './temp/'
        self.optim_num_steps = [1]
        self.text = "van gogh style"
        self.trials = 10
        self.samples_per_diffusion = 4
        
        self.progressive = True


class DanQ(nn.Module):
    def __init__(self, ):
        super(DanQ, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=26)
        #self.Conv1.weight.data = torch.Tensor(np.load('conv1_weights.npy'))
        #self.Conv1.bias.data = torch.Tensor(np.load('conv1_bias.npy'))
        self.Maxpool = nn.MaxPool1d(kernel_size=13, stride=13)
        self.Drop1 = nn.Dropout(p=0.2)
        self.BiLSTM = nn.LSTM(input_size=320, hidden_size=320, num_layers=2,
                                 batch_first=True,
                                 dropout=0.5,
                                 bidirectional=True)
        self.Linear1 = nn.Linear(13*640, 925)
        self.Linear2 = nn.Linear(925, NUM_CLASS)

    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x_x = torch.transpose(x, 1, 2)
        x, (h_n,h_c) = self.BiLSTM(x_x)
        #x, h_n = self.BiGRU(x_x)
        x = x.contiguous().view(-1, 13*640)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        return x




