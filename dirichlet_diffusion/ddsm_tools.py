import os
import numpy as np
import pandas as pd

#from pyjaspar import jaspardb
import pychromvar as pc

import torch
from torch import nn
# from torch.utils.data import DataLoader, Dataset
# from torch.optim import Adam

import time
import tqdm
#import tabix
#import pyBigWig
import pandas as pd
from matplotlib import pyplot as plt

import pickle
import sys
import os
from collections import Counter

file_dir = os.path.dirname(__file__)
data_dir = os.path.join(file_dir, '..','data_utils','generated_data')
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
    UnitStickBreakingTransform,
    Euler_Maruyama_sampler
    )
from utils import extract_motifs

sb = UnitStickBreakingTransform()

class TrainDDSM:
    def __init__(self, config, data, sampler):
        self.config = config
        self.sampler = sampler
        self.seqlen = config.seq_length
        v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints = torch.load(config.diffusion_weights_file)
        self.v_one, self.v_zero, self.v_one_loggrad, self.v_zero_loggrad, self.timepoints = v_one.cpu(), v_zero.cpu(), v_one_loggrad.cpu(), v_zero_loggrad.cpu(), timepoints.cpu()
        
        self.alpha = torch.ones(config.ncat - 1).float()
        self.beta =  torch.arange(config.ncat - 1, 0, -1).float()
        self.n_time_steps = timepoints.shape[0]
        self.min_time = timepoints[0].item()
        self.max_time = timepoints[-1].item()

        
        self.cell_types=data["cell_types"]
        # count cell types in train
        cell_dict_temp = Counter(data['x_train_cell_type'].tolist())
        # reoder by cell_types list
        cell_dict = {k:cell_dict_temp[k] for k in data['cell_types']}
        # take only counts
        cell_type_counts = list(cell_dict.values())
        self.cell_type_probabilities = [x / sum(cell_type_counts) for x in cell_type_counts]

    
    def init_train(self, score_model, optimizer):
        self.score_model = score_model
        self.score_model.train()
        self.optimizer = optimizer
    
    def get_time_dependent_weights(self, file_path, data_loader, load_saved_file=False):
        config = self.config
        v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints = self.v_one, self.v_zero, self.v_one_loggrad, self.v_zero_loggrad, self.timepoints
        time_dependent_cums = torch.zeros(config.n_time_steps).to(config.device)
        time_dependent_counts = torch.zeros(config.n_time_steps).to(config.device)
        if load_saved_file:
            with open(file_path, 'rb') as file:
                time_dependent_weights = pickle.load(file)
        else: 
            for i, x in enumerate(data_loader):
                print(f'precomputing weights {i}', end='\r')
                x = x[..., :4]
                random_t = torch.randint(0, config.n_time_steps, (x.shape[0],))

                order = np.random.permutation(np.arange(config.ncat))
                # There are two options for using presampled noise: 
                # First one is regular approach and second one is fast sampling (see Appendix A.4 for more info)
                perturbed_x, perturbed_x_grad = diffusion_factory(x, random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad,
                                                                    self.alpha, self.beta)
                # perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x, random_t, v_one, v_one_loggrad)
                
                perturbed_x = perturbed_x.to(config.device)
                perturbed_x_grad = perturbed_x_grad.to(config.device)
                random_t = random_t.to(config.device)
                perturbed_v = sb._inverse(perturbed_x)

                order = np.random.permutation(np.arange(config.ncat))

                perturbed_v = sb._inverse(perturbed_x, prevent_nan=True).detach()

                time_dependent_counts[random_t] += 1
                if config.speed_balanced:
                    s = 2 / (torch.ones(config.ncat - 1, device=config.device) + torch.arange(config.ncat - 1, 0, -1,
                                                                                                device=config.device).float())
                else:
                    s = torch.ones(config.ncat - 1, device=config.device)

                time_dependent_cums[random_t] += (perturbed_v * (1 - perturbed_v) * s[(None,) * (x.ndim - 1)] * (
                    gx_to_gv(perturbed_x_grad, perturbed_x)) ** 2).view(x.shape[0], -1).mean(dim=1).detach()

            time_dependent_weights = time_dependent_cums / time_dependent_counts
            time_dependent_weights = time_dependent_weights / time_dependent_weights.mean()

            plt.plot(torch.sqrt(time_dependent_weights.cpu()))
            plt.savefig(f"{save_dir}/timedependent_weight.png")

            with open(file_path, 'wb') as file:
                pickle.dump(time_dependent_weights, file)
        print("loaded weights")
        self.time_dependent_weights = time_dependent_weights
        return time_dependent_weights

    def ddsm_train_step(self, x,s, avg_loss, num_items):
        config = self.config
        v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints = self.v_one, self.v_zero, self.v_one_loggrad, self.v_zero_loggrad, self.timepoints
        # Optional : there are several options for importance sampling here. it needs to match the loss function
        random_t = torch.LongTensor(np.random.choice(np.arange(config.n_time_steps), size=x.shape[0],
                                                        p=(torch.sqrt(self.time_dependent_weights) / torch.sqrt(
                                                            self.time_dependent_weights).sum()).cpu().detach().numpy()))

        # There are two options for using presampled noise: 
        # First one is regular approach and second one is fast sampling (see Appendix A.4 for more info)
        # perturbed_x, perturbed_x_grad = diffusion_factory(x, random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, self.alpha, self.beta)
        perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x.cpu(), random_t, v_one, v_one_loggrad)
            

        perturbed_x = perturbed_x.to(config.device)
        perturbed_x_grad = perturbed_x_grad.to(config.device)
        random_timepoints = timepoints[random_t].to(config.device)

        # random_t = random_t.to(config.device)

        s = s.to(config.device)

        score = self.score_model(torch.cat([perturbed_x, s], -1), random_timepoints)

        # the loss weighting function may change, there are a few options that we will experiment on
        if config.speed_balanced:
            s = 2 / (torch.ones(config.ncat - 1, device=config.device) + torch.arange(config.ncat - 1, 0, -1,
                                                                                        device=config.device).float())
        else:
            s = torch.ones(config.ncat - 1, device=config.device)

        
        perturbed_v = sb._inverse(perturbed_x, prevent_nan=True).detach()
        loss = torch.mean(torch.mean(
            1 / (torch.sqrt(self.time_dependent_weights))[random_t][(...,) + (None,) * (x.ndim - 1)] * s[
                (None,) * (x.ndim - 1)] * perturbed_v * (1 - perturbed_v) * (
                        gx_to_gv(score, perturbed_x, create_graph=True) - gx_to_gv(perturbed_x_grad,
                                                                                    perturbed_x)) ** 2, dim=(1)))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
        return avg_loss, num_items
    
    def create_sample(self):
        self.score_model.eval()
        nucleotides = ["A", "C", "G", "T"]
        number_of_samples = self.config.num_sampling_to_compare_cells
        sample_bs = self.config.sample_bs
        device = self.config.device
        plain_generated_sequences = []
        final_sequences = []
            
        for n_a in range(int(number_of_samples/ sample_bs)):
            sampled_cell_types = np.random.choice(self.cell_types, sample_bs, p=self.cell_type_probabilities)
            sampled = np.repeat(sampled_cell_types[:, np.newaxis], self.seqlen, axis=1).reshape(sample_bs, self.seqlen, 1)
            classes = torch.from_numpy(sampled).float().to(device)
            samples = self.sampler(self.score_model,
                            (self.seqlen, 4),
                            batch_size=sample_bs,
                            max_time=self.max_time,
                            min_time=self.min_time,
                            time_dilation=4,
                            time_dilation_start_time=1,
                            num_steps=400,
                            eps=1e-5,
                            device=device,
                            speed_balanced = self.config.speed_balanced,
                            concat_input = classes)

            sampled_images = samples.clamp(0.0, 1.0)
            
            for n_b, x in enumerate(sampled_images):
                sequence = "".join([nucleotides[s] for s in np.argmax(x.detach().cpu(), axis=1)])
                plain_generated_sequences.append(sequence)
                seq_final = f">seq_train_{n_a}_{n_b}\n" + sequence
                final_sequences.append(seq_final)
                
        df_motifs_count_syn = extract_motifs(final_sequences)
        self.score_model.train()
        return df_motifs_count_syn, plain_generated_sequences


class Dense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[...]


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, embed_dim=256, time_dependent_weights=None, time_step=0.01):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
        # build a CNN that does 1d convolutions over the input sequence
        n = 256
        self.linear = nn.Conv1d(5, n, kernel_size=9, padding=4)
        self.blocks = nn.ModuleList([nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256)])

        self.denses = nn.ModuleList([Dense(embed_dim, n) for _ in range(20)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, n) for _ in range(20)])

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.scale = nn.Parameter(torch.ones(1))
        self.final = nn.Sequential(nn.Conv1d(n, n, kernel_size=1),
                                   nn.GELU(),
                                   nn.Conv1d(n, 4, kernel_size=1))
        self.register_buffer("time_dependent_weights", time_dependent_weights)
        self.time_step = time_step

    def forward(self, x, t, t_ind=None, return_a=False):
        # Obtain the Gaussian random feature embedding for t
        # embed: [N, embed_dim]
        embed = self.act(self.embed(t / 2))

        # Encoding path
        # x: NLC -> NCL
        out = x.permute(0, 2, 1)
        out = self.act(self.linear(out))

        # pos encoding
        # the zip creates an iterator passes the embedding through N consecutive blocks of convolution->linear->norm
        for block, dense, norm in zip(self.blocks, self.denses, self.norms):
            # embed every base in the sequence with the same n (eg. 256) dimensional time embedding: "out + dense(embed)[:, :, None]"
            h = self.act(block(norm(out + dense(embed)[:, :, None])))
            if h.shape == out.shape:
                out = h + out
            else:
                out = h

        out = self.final(out)

        out = out.permute(0, 2, 1)

        if self.time_dependent_weights is not None:
            t_step = (t / self.time_step) - 1
            w0 = self.time_dependent_weights[t_step.long()]
            w1 = self.time_dependent_weights[torch.clip(t_step + 1, max=len(self.time_dependent_weights) - 1).long()]
            out = out * (w0 + (t_step - t_step.floor()) * (w1 - w0))[:, None, None]

        out = out - out.mean(axis=-1, keepdims=True)
        return out

    