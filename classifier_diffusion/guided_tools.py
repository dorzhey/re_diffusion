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

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.

file_dir = os.path.dirname(__file__)
data_dir = os.path.join(file_dir, '..','data_preprocessing','generated_data')
checkpoints_dir =  os.path.join(file_dir,'diffusion_checkpoints')

guided_diff_path = os.path.join(file_dir, '..', '..','re_design', 'guided-diffusion')
if guided_diff_path not in sys.path:
    sys.path.append(guided_diff_path)

from guided_diffusion.unet import UNetModel #, EncoderUNetModel
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.nn import update_ema
from guided_diffusion.resample import LossAwareSampler, UniformSampler

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






NUM_CLASSES = 3

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


def create_classifier(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
):
    # if image_size == 512:
    #     channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    # elif image_size == 256:
    #     channel_mult = (1, 1, 2, 2, 4, 4)
    # elif image_size == 128:
    #     channel_mult = (1, 1, 2, 3, 4)
    # elif image_size == 64:
    #     channel_mult = (1, 2, 3, 4)
    # else:
    #     raise ValueError(f"unsupported image size: {image_size}")

    channel_mult = (1, 2, 3)
    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=1,
        model_channels=classifier_width,
        out_channels=NUM_CLASSES,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        dims=2,
        use_fp16=classifier_use_fp16,
        num_head_channels=4,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )

def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    # if channel_mult == "":
    #     if image_size == 512:
    #         channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    #     elif image_size == 256:
    #         channel_mult = (1, 1, 2, 2, 4, 4)
    #     elif image_size == 128:
    #         channel_mult = (1, 1, 2, 3, 4)
    #     elif image_size == 64:
    #         channel_mult = (1, 2, 3, 4)
    #     else:
    #         raise ValueError(f"unsupported image size: {image_size}")
    # else:
    #     channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
    channel_mult = (1, 2, 3)
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=1,
        model_channels=num_channels,
        out_channels=(1 if not learn_sigma else 3),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        dims=2,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )



class TrainLoop:
    def __init__(
        self,
        *,
        config,
        model,
        diffusion,
        classifier,
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
        self.classifier = classifier
        self.config = config
        
        # data["X_train"][data["X_train"] == -1] = 0
        self.data = data
        seq_dataset = SequenceDataset(seqs=data["X_train"], c=data["x_train_cell_type"], transform_ddsm = False) 
        self.seq_dataset = SequenceDataset(seqs=data["X_train"], c=data["x_train_cell_type"], transform_ddsm = False)
        self.seq_dataset.seqs = seq_dataset.seqs.astype(np.float16 if config.classifier_use_fp16 else np.float32)
        self.seq_dataset.c = seq_dataset.c.long()
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
    
    def create_sample(self):
        self.model.eval()
        nucleotides = ["A", "C", "G", "T"]
        final_sequences = []
        plain_generated_sequences = []
        for n_a in range(int(self.config.num_sampling_to_compare_cells/ self.config.sample_bs)):
            model_kwargs = {}
            if self.config.class_cond:
                classes = torch.from_numpy(np.random.choice(self.data['cell_types'], self.config.sample_bs, p=self.cell_type_probabilities)).to(dist_util.dev())
                model_kwargs["y"] = classes
            sample_fn = (
                self.diffusion.p_sample_loop if not self.config.use_ddim else self.diffusion.ddim_sample_loop
            )
            sampled_images = sample_fn(
                self.model_fn if self.config.use_classifier else self.model,
                (self.config.sample_bs, 1, 4, self.config.image_size),
                clip_denoised=self.config.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn = self.cond_fn if self.config.use_classifier else None,
                device=dist_util.dev()
            ).squeeze(1)
            
            for n_b, x in enumerate(sampled_images):
                sequence = "".join([nucleotides[s] for s in np.argmax(x.detach().cpu(), axis=0)])
                plain_generated_sequences.append(sequence)
                seq_final = f">seq_test_{n_a}_{n_b}\n" + sequence
                final_sequences.append(seq_final)
        df_motifs_count_syn = extract_motifs(final_sequences)
        self.model.train()
        return df_motifs_count_syn, plain_generated_sequences

    def create_sample_labelwise(self):
        self.model.eval()
        nucleotides = ["A", "C", "G", "T"]
        generated_celltype_motif = {}
        generated_celltype_sequences = {}
        bulk_final_sequences = []
        for cell_num in self.cell_num_list:
            cell_type = self.data['numeric_to_tag'][cell_num]
            final_sequences = []
            plain_generated_sequences = []
            cell_type_sample_size = int(self.cell_num_sample_counts[cell_num] / self.config.sample_bs)
            print(f"Generating {int(self.cell_num_sample_counts[cell_num]) // self.config.sample_bs * self.config.sample_bs} samples for cell_type {cell_type}")
            for n_a in range(cell_type_sample_size):
                model_kwargs = {}
                sampled_cell_types = np.array([cell_num] * self.config.sample_bs)
                classes = torch.from_numpy(sampled_cell_types).to(dist_util.dev())
                model_kwargs["y"] = classes
                sample_fn = (
                    self.diffusion.p_sample_loop if not self.config.use_ddim else self.diffusion.ddim_sample_loop
                )
                sampled_images = sample_fn(
                    self.model_fn if self.config.use_classifier else self.model,
                    (self.config.sample_bs, 1, 4, self.config.image_size),
                    clip_denoised=self.config.clip_denoised,
                    model_kwargs=model_kwargs,
                    cond_fn = self.cond_fn if self.config.use_classifier else None,
                    device=dist_util.dev()
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
            logits = self.classifier(x_in, t)
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
    

import torch.nn as nn
import torch.nn.functional as F
from guided_diffusion.unet import (
    Downsample, 
    AttentionBlock, 
    AttentionPool2d, 
    linear, 
    TimestepEmbedSequential, 
    conv_nd,
    normalization,
    zero_module,
    convert_module_to_f16,
    timestep_embedding,
    convert_module_to_f32,
    Upsample,
    TimestepBlock,
    checkpoint,
)

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, kernel_size = (3,16), stride=1, padding=(1, 15)))] 
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # my code:
        #self._feature_size = 200

        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        # print("feature_size ", self._feature_size)
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                # their comment: CHANGED dim=(2, 3) to dim=(1, 2)
                results.append(h.type(x.dtype).mean(dim=(2,3))) # I CHANGED HERE, from 2 to 2,3
        #     print(h.size())
        # print("middle")
        h = self.middle_block(h, emb)
        # print(h.size())
        # print("more")
        if self.pool.startswith("spatial"):
            # their comment: CHANGED dim=(2, 3) to dim=(1, 2)
            results.append(h.type(x.dtype).mean(dim=(2,3))) # I CHANGED HERE, from 2 to 2,3
            # for temp in results:
            #     print(temp.size())
            # my added code:
            # the problem was that square image will result in format that will fit in torch.cat, whereas my data has a dimension of size 4, so I had to decrease the channel_mult from (1,2,3,4) to (1,2,3) which caused errors
            # so I tried to fit it all together following the initial logic as I close as I can
            h = th.cat(results, dim=1) # I CHANGED HERE FROM 2 TO 1
            # their code
            # print("spatial:")
            # print(h.size())

            return self.out(h)
        else:
            h = h.type(x.dtype)
            # print(h.size())
            return self.out(h)
        

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=1,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            #print(h.size())
            hs.append(h)
        #print("midlle")
        h = self.middle_block(h, emb)
        #print(h.size())
        #print("output")
        for module in self.output_blocks:
            #print(h.size())
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
            # if h.size()[2] != hs[-1].size()[2]:
            #     print("error")
            #     print(module)
        h = h.type(x.dtype)
        return self.out(h)
    

def timestep_embedding_danq(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    """
    half_dim = dim // 2
    emb = torch.log(torch.tensor(max_period)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).to(device=timesteps.device)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

    
class DanQ(nn.Module):
    def __init__(self):
        super(DanQ, self).__init__()
        self.model_channels = 320
        self.time_embed_dim = self.model_channels * 4
        
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=26)
        self.Maxpool = nn.MaxPool1d(kernel_size=13, stride=13)
        self.Drop1 = nn.Dropout(p=0.2)
        self.BiLSTM = nn.LSTM(input_size=320, hidden_size=320, num_layers=2,
                              batch_first=True,
                              dropout=0.5,
                              bidirectional=True)
        self.Linear1 = nn.Linear(13*640, 925)
        self.Linear2 = nn.Linear(925, NUM_CLASSES)
        
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        
        self.project_time_embed = nn.Linear(self.time_embed_dim, 320)

    def forward(self, input, timesteps):
        # Generate timestep embeddings
        emb = self.time_embed(timestep_embedding_danq(timesteps, self.model_channels))
        
        # Project the timestep embeddings to match the conv output channels
        emb = self.project_time_embed(emb)
        
        # Pass input through convolutional layer
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        
        # Ensure timestep embeddings match the shape of the conv output
        emb = emb.unsqueeze(2).expand(-1, -1, x.shape[-1])
        x = x + emb

        # Transpose and pass through LSTM
        x = torch.transpose(x, 1, 2)
        x, (h_n, h_c) = self.BiLSTM(x)
        # print(x.size())
        # Flatten and pass through fully connected layers
        x = x.contiguous().view(-1, 13*640)
        # print(x.size())
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        # print(x.size())
        return x