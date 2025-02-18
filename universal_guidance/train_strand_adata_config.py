import os
import pyfaidx
import pathlib
import time
import random
import wandb
import re
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

import pickle
import json

from goatools.base import download_go_basic_obo
from goatools.obo_parser import GODag

import datetime

from dragonnfruit.preprocessing import *
from dragonnfruit.models import *
from dragonnfruit.io import *
from dragonnfruit.performance_metrics import pearson_corr, multinomial_log_probs, compute_performance_metrics

from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, lil_matrix
from scipy.spatial.distance import pdist, squareform

import torch.optim.lr_scheduler
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torch

import collections

import subprocess

from bpnetlite.io import one_hot_encode
from bpnetlite.losses import MNLLLoss, log1pMSELoss, DMNLLLoss, MNLLLossOld
from bpnetlite.performance import pearson_corr
from bpnetlite.logging import Logger

from filelock import FileLock

from tqdm import tqdm
import bz2
import scanpy as sc
import gzip
import mudata as md

os.environ['USE_CELLS'] = 'centroid_res20'
os.environ['BIAS_FEATURE_EMBEDDING'] = 'pca'
os.environ['BIAS_FEATURES'] = 'tf'
# os.environ['BIAS_FEATURES'] = 'all_genes'
os.environ['DEBUG_STRAND'] = '0'
os.environ['COUNTS_ALPHA'] = '0.1'
os.environ['N_NEIGHBORS_SC'] = '200'
os.environ['MAX_EPOCHS'] = '5000'
os.environ['WINDOW_LENGTH'] = '2048'
os.environ['MAX_JITTER'] = '2'
os.environ['TRIMMING'] = '512'
os.environ['VALIDATION_FREQ'] = '100'
os.environ['LR'] = '0.001'
os.environ['SCHED_METRIC'] = 'val'
os.environ['VAL_BATCH_SIZE'] = '32'
os.environ['TRAIN_BATCH_SIZE'] = '64'
os.environ['RSEED'] = '42'
os.environ['NUM_WORKERS'] = '32'
os.environ['BP_FILTERS'] = '512'
os.environ['CTRL_LAYERS'] = '1'
os.environ['CTRL_NODES'] = '256'
os.environ['CTRL_OUTPUTS'] = '64'
os.environ['SCHED_PATIENCE'] = '48'
os.environ['SCHED_FACTOR'] = '0.1'
os.environ['SCHED_THRESH'] = '1e-3'
# os.environ['WANDB_PROJ_NAME'] = 'bp_debug_generator_new_window_all_obs'
os.environ['WANDB_PROJ_NAME'] = 'debug_gut'
os.environ['WANDB_NOTEBOOK_NAME'] = 'start5prime_debug_gl_multi'
# os.environ['RUN_DATA_DIR'] = '/scratch/sigbio_project_root/sigbio_project25/mkarikom/runs_debug'
# os.environ['RUN_DATA_DIR'] = '/scratch/welchjd_root/welchjd5/mkarikom/_debug'
# os.environ['RUN_DATA_DIR'] = '/workspaces/torch_ddsm/_torch_save_completion'
# os.environ['RUN_DATA_DIR'] = '/workspaces/torch_ddsm/_data_pool1/_torch_save_frag_profile'
os.environ['RUN_DATA_DIR'] = '/scratch/welchjd_root/welchjd5/mkarikom/scBPNet_runs'
# os.environ['SUBSET_CHROMS'] = 'chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrM,chrX,chrY'
os.environ['SUBSET_CHROMS'] = 'chr10,chr11,chr15,chr16'
os.environ['TEST_CHROMS'] = 'chr15,chr16'
os.environ['VALIDATION_STEP_UNIT'] = 'batch'
os.environ['MODEL_VARIANT'] = 'regularized_counts'
os.environ['DROPOUT_RATE'] = '0.2'
os.environ['ADATA_FILE'] = '/scratch/welchjd_root/welchjd5/mkarikom/scBPNet_prepro/adata_fetal_sub.h5ad'
os.environ['GENOME_REF'] = '/nfs/turbo/umms-welchjd/mkarikom/SCAFE/resources/genome/hg38.gencode_v32/fasta/genome.fa'
os.environ['BC_FILES'] = '/scratch/welchjd_root/welchjd5/mkarikom/scBPNet_prepro/batch_files_fetal'
os.environ['SIGNALS_DIR'] = "/scratch/welchjd_root/welchjd5/mkarikom/E-MTAB-9536_processed"
# os.environ['BATCH_SUBSET']= "FCA_gut8015057,FCA_gut8015058,FCA_gut8015059,FCA_gut8015060,FCA_gut8015061"
os.environ['BATCH_SUBSET']= "FCA_gut8015058"
os.environ['LOCUS_BED_FILE'] = '/scratch/welchjd_root/welchjd5/mkarikom/E-MTAB-9536_processed/FCA_gut8015058/scafe/directionality/filtered/bed/filtered.CRE.directionality.summit_center.bed.gz'

def merge_parameters(parameters, default_parameters):
	"""Merge the provided parameters with the default parameters.

	
	Parameters
	----------
	parameters: str
		Name of the JSON folder with the provided parameters

	default_parameters: dict
		The default parameters for the operation.


	Returns
	-------
	params: dict
		The merged set of parameters.
	"""

	with open(parameters, "r") as infile:
		parameters = json.load(infile)

	for parameter, value in default_parameters.items():
		if parameter not in parameters:
			parameters[parameter] = value

	return parameters

def get_barcode_map(barcode_path):
	barcode_map = {}
	with open(barcode_path, 'r') as f:
		for i, line in enumerate(f):
			barcode_map[line.strip()] = i
	return barcode_map

# def get_barcode_map(barcode_path):
# 	barcode_map = {}
# 	# Open the file depending on the extension
# 	if barcode_path.endswith('.gz'):
# 		with gzip.open(barcode_path, 'rt') as f:  # 'rt' mode for text reading
# 			for i, line in enumerate(f):
# 				barcode_map[line.strip()] = i
# 	else:
# 		with open(barcode_path, 'r') as f:
# 			for i, line in enumerate(f):
# 				barcode_map[line.strip()] = i
# 	return barcode_map

def load_signals(data_dir,subset_bcs=None, chroms=None):
	# Initialize the dictionary to store coo_matrix for each chromosome and strand
	chrom_matrices = {'pos':collections.defaultdict(coo_matrix),'neg':collections.defaultdict(coo_matrix)}

	# Pattern to match filenames ending with .pkl or .pkl.bz2
	pattern = r"profile_(chr\d+)_[+-]\.pkl(\.bz2)?"

	for filename in os.listdir(data_dir):
		if filename.startswith('profile_chr') and (filename.endswith('.pkl') or filename.endswith('.pkl.bz2')):
			match = re.match(pattern, filename)
			if match is not None:
				if ((chroms is not None) and (match.group(1) in chroms)) or (chroms is None):
					file_path = os.path.join(data_dir, filename)
					chrom, strand = filename.split('.')[0].split('_')[1:]
					
					if filename.endswith('.pkl.bz2'):
						with bz2.open(file_path, 'rb') as f:
							coo_mat = pickle.load(f)
					else:
						with open(file_path, 'rb') as f:
							coo_mat = pickle.load(f)
					
					csc_mat = coo_mat.tocsc()
					if subset_bcs is not None:
						csc_mat = csc_mat[subset_bcs, :]
						
					if strand == '+':
						chrom_matrices['pos'][chrom] = csc_mat
						print(f"{chrom}{strand} min profile value is {csc_mat.min()}, max profile value is {csc_mat.max()}")
					elif strand == '-':
						chrom_matrices['neg'][chrom] = csc_mat
						print(f"{chrom}{strand} min profile value is {csc_mat.min()}, max profile value is {csc_mat.max()}")
					del csc_mat

	return chrom_matrices

def get_read_depths(signals):
	# calculates the cell,strand-wise read-depth across all chromosomes
	# returns a dictionary with keys 'pos' and 'neg' and values are numpy arrays of shape (n_cells,) 
	read_depths_pos = []
	read_depths_neg = []
	for chrom_pos, signal_pos in signals['pos'].items():
		read_depths_pos.append(signal_pos.sum(axis=1))
	read_depths_pos = np.concatenate(read_depths_pos,axis=1,dtype=np.int64)
	for chrom_neg, signal_neg in signals['neg'].items():
		read_depths_neg.append(signal_neg.sum(axis=1))
	read_depths_neg = np.concatenate(read_depths_neg,axis=1,dtype=np.int64)
	return {'pos':np.array(read_depths_pos.sum(axis=1)), 'neg':np.array(read_depths_neg.sum(axis=1))}

def get_neighbors(dist_mat_orig,k=50):
	dist_mat = dist_mat_orig.toarray()    
	all_sorted_inds = []
	all_sorted_dists = []
	k_sorted_inds = []
	k_sorted_dists = []
	for i in range(dist_mat.shape[0]):
		all_neighbors = np.nonzero(dist_mat[i,:])[0]
		sorted_ind = all_neighbors[np.argsort(dist_mat[i,all_neighbors])]
		sorted_dist = dist_mat[i,sorted_ind]
		all_sorted_inds.append(sorted_ind)
		all_sorted_dists.append(sorted_dist)
		k_sorted_inds.append(sorted_ind[:k])
		k_sorted_dists.append(sorted_dist[:k])
	return all_sorted_dists, all_sorted_inds, k_sorted_dists, k_sorted_inds

def get_smoothed_cell_state(cell_ind, cell_neighbors, all_cell_states):
	neighbor_inds = cell_neighbors[cell_ind]
	smoothed_cell_state = np.sum(all_cell_states[neighbor_inds,:],axis=0)/len(neighbor_inds)
	return np.array(smoothed_cell_state)[np.newaxis,:]

def get_smoothed_cell_states(cell_neighbors, all_cell_states):
	all_smoothed_states = []
	for i in range(len(all_sorted_inds)):
		all_smoothed_states.append(get_smoothed_cell_state(i, cell_neighbors,all_cell_states))
	all_smoothed_states = np.concatenate(all_smoothed_states,axis=0)
	return all_smoothed_states

class RegularizedDynamicBPNet(torch.nn.Module):

	def __init__(self, controller, n_filters=128, n_layers=8, trimming=None, 
		conv_bias=False, n_outputs=1, dropout_rate=0.2):
		super(RegularizedDynamicBPNet, self).__init__()

		self.trimming = trimming if trimming is not None else 2 ** n_layers + 37
		self.n_filters = n_filters
		self.n_layers = n_layers
		self.n_outputs = n_outputs
		self.dropout_rate = dropout_rate
		self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)
		self.irelu = torch.nn.ReLU()
		self.idropout = torch.nn.Dropout(p=dropout_rate)
		self.fconv = torch.nn.Conv1d(n_filters, self.n_outputs, kernel_size=75, padding=37,
			bias=conv_bias)

		self.biases = torch.nn.ModuleList([
			torch.nn.Linear(controller.n_outputs, n_filters) for i in range(
				n_layers)
		])

		self.convs = torch.nn.ModuleList([
			torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, stride=1, 
				dilation=2**i, padding=2**i, bias=conv_bias) for i in range(1, 
					n_layers+1)
		])

		self.relus = torch.nn.ModuleList([
			torch.nn.ReLU() for i in range(n_layers)
		])

		self.dropouts = torch.nn.ModuleList([
			torch.nn.Dropout(p=dropout_rate) for _ in range(n_layers)
		])
	   
		self.controller = controller

	def forward(self, X, cell_states):

		start, end = self.trimming, X.shape[2] - self.trimming
		
		cell_states = self.controller(cell_states)
		X = self.irelu(self.iconv(X))
		X = self.idropout(X)

		for i in range(self.n_layers):
			X_conv = self.convs[i](X)
			X_bias = self.biases[i](cell_states).unsqueeze(-1)			
			X = X + self.relus[i](X_conv + X_bias)
			X = self.dropouts[i](X)

		y_profile = self.fconv(X)[:, :, start:end]
		return y_profile


class RegularizedDynamicBPNetCounts(torch.nn.Module):

	def __init__(self, controller, n_filters=128, n_layers=8, trimming=None, 
		conv_bias=False, n_outputs=1, dropout_rate=0.2):
		super(RegularizedDynamicBPNetCounts, self).__init__()

		self.trimming = trimming if trimming is not None else 2 ** n_layers + 37
		self.n_filters = n_filters
		self.n_layers = n_layers
		self.n_outputs = n_outputs
		self.dropout_rate = dropout_rate
		self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)
		self.irelu = torch.nn.ReLU()
		self.idropout = torch.nn.Dropout(p=dropout_rate)
  		
		self.deconv_kernel_size = 75
		self.fconv = torch.nn.Conv1d(self.n_filters, self.n_outputs, kernel_size=self.deconv_kernel_size,
			bias=conv_bias)

		self.biases = torch.nn.ModuleList([
			torch.nn.Linear(controller.n_outputs, n_filters) for i in range(
				n_layers)
		])

		self.convs = torch.nn.ModuleList([
			torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, stride=1, 
				dilation=2**i, padding=2**i, bias=conv_bias) for i in range(1, 
					n_layers+1)
		])

		self.relus = torch.nn.ModuleList([
			torch.nn.ReLU() for i in range(n_layers)
		])

		self.linear = torch.nn.Linear(n_filters, 1)

		self.dropouts = torch.nn.ModuleList([
			torch.nn.Dropout(p=dropout_rate) for _ in range(n_layers)
		])
	   
		self.controller = controller

	def forward(self, X, cell_states):

		start, end = self.trimming, X.shape[2] - self.trimming
		cell_states = self.controller(cell_states)
		X = self.irelu(self.iconv(X))
		X = self.idropout(X)
		for i in range(self.n_layers):
			X_conv = self.convs[i](X)
			X_bias = self.biases[i](cell_states).unsqueeze(-1)			
			X = X + self.relus[i](X_conv + X_bias)
			X = self.dropouts[i](X)

		X = X[:, :, start - self.deconv_kernel_size//2 : end + self.deconv_kernel_size//2]
		y_profile = self.fconv(X)
		X = torch.mean(X, axis=2)
		y_counts = self.linear(X)

		return y_profile, y_counts



class scBPnetCounts(DragoNNFruit):
	def __init__(self, accessibility, name, alpha=1, scale_log_rd=False, n_outputs=2):
		torch.nn.Module.__init__(self)
		self.accessibility = accessibility
		self.name = name
		self.alpha = alpha
		self.n_outputs = n_outputs
		self.scale_log_rd = scale_log_rd
		self.logger = Logger(["Epoch", "Iteration", "Training Time",
			"Validation Time", "Training MNLL", "Validation MNLL",
			"Validation Profile Correlation", "Validation Count Correlation", 
			"Saved?"], verbose=True)

	def forward(self, X, cell_states):

		return self.accessibility(X, cell_states)

	def log_softmax(self, y_profile):
		y_profile = y_profile.reshape(y_profile.shape[0], -1)
		y_profile = torch.nn.LogSoftmax(dim=-1)(y_profile)
		y_profile = y_profile.reshape(y_profile.shape[0], self.n_outputs, -1)
		return y_profile
	
	def predict(self, X, cell_states, batch_size=64, logits = False):
		with torch.no_grad():
			starts = np.arange(0, X.shape[0], batch_size)

			ends = starts + batch_size

			y_profiles, y_counts = [], []
			for start, end in zip(starts, ends):
				X_batch = X[start:end]
				cell_states_batch = cell_states[start:end]
				y_profiles_, y_counts_ = self(X_batch,cell_states_batch)
				if not logits:  # apply softmax
					y_profiles_ = self.log_softmax(y_profiles_)
				y_profiles.append(y_profiles_.cpu().detach().numpy())
				y_counts.append(y_counts_.cpu().detach().numpy())

			y_profiles = np.concatenate(y_profiles)
			y_counts = np.concatenate(y_counts)
			return y_profiles, y_counts


class scBPnet(DragoNNFruit):
	def __init__(self, accessibility, name, alpha=1, scale_log_rd=False):
		# super().__init__(accessibility, name, alpha)
		torch.nn.Module.__init__(self)
		self.accessibility = accessibility
		self.name = name
		self.alpha = alpha
		self.scale_log_rd = scale_log_rd
		self.logger = Logger(["Epoch", "Iteration", "Training Time",
			"Validation Time", "Training MNLL", "Validation MNLL",
			"Validation Profile Correlation", "Validation Count Correlation", 
			"Saved?"], verbose=True)

	def forward(self, X, cell_states, read_depths):
		accessibility = self.accessibility(X, cell_states)

		return read_depths.unsqueeze(1) + accessibility

	@torch.no_grad()
	def predict(self, X, cell_states, read_depths, batch_size=16, 
		reduction=None, verbose=False):

		y_hat = []
		for start in trange(0, len(X), batch_size, disable=not verbose):
			X_batch = X[start:start+batch_size]
			cs_batch = cell_states[start:start+batch_size]
			rd_batch = read_depths[start:start+batch_size]

			y_hat_ = self(X_batch, cs_batch, rd_batch)
			if reduction == 'sum':
				y_hat_ = torch.sum(y_hat_, dim=-1)
			elif reduction == 'logsumexp':
				y_hat_ = torch.logsumexp(y_hat_, dim=-1)

			y_hat.append(y_hat_)

		y_hat = torch.cat(y_hat)
		return y_hat

def get_random_indices(lst, n, random_state):
	if n == 0:
		return lst
	else:
		return random_state.choice(lst, min(len(lst), n), replace=False)

use_cells = os.environ.get('USE_CELLS')
bias_feature_embedding = os.environ.get('BIAS_FEATURE_EMBEDDING')
bias_features = os.environ.get('BIAS_FEATURES')
debug_strand = int(os.environ.get('DEBUG_STRAND'))
n_neighbors_sc = int(os.environ.get('N_NEIGHBORS_SC'))
max_epochs = int(os.environ.get('MAX_EPOCHS'))
window_length = int(os.environ.get('WINDOW_LENGTH'))
max_jitter = int(os.environ.get('MAX_JITTER'))
trimming = int(os.environ.get('TRIMMING'))
validation_freq = int(os.environ.get('VALIDATION_FREQ'))
lr = float(os.environ.get('LR'))
counts_alpha = float(os.environ.get('COUNTS_ALPHA'))
train_batch_size = int(os.environ.get('TRAIN_BATCH_SIZE'))
val_batch_size = int(os.environ.get('VAL_BATCH_SIZE'))
rseed = int(os.environ.get('RSEED'))
num_workers = int(os.environ.get('NUM_WORKERS'))
bp_filters = int(os.environ.get('BP_FILTERS'))
ctrl_layers = int(os.environ.get('CTRL_LAYERS'))
ctrl_nodes = int(os.environ.get('CTRL_NODES'))
ctrl_outputs = int(os.environ.get('CTRL_OUTPUTS'))
sched_patience = int(os.environ.get('SCHED_PATIENCE'))
sched_factor = float(os.environ.get('SCHED_FACTOR'))
sched_thresh = float(os.environ.get('SCHED_THRESH'))
run_data_dir = os.environ.get('RUN_DATA_DIR')
sched_metric = os.environ.get('SCHED_METRIC')
wandb_proj_name = os.environ.get('WANDB_PROJ_NAME')
current_time = datetime.datetime.now()
save_model_time = current_time.strftime("%d%b%Y_%H.%M.%S.%f")
conv_layers = np.log2(window_length).astype(int) - 1
subset_chroms_env = os.environ.get('SUBSET_CHROMS')
bc_files = os.environ.get('BC_FILES')
signals_dir = os.environ.get('SIGNALS_DIR')
test_chroms_env = os.environ.get('TEST_CHROMS')
validation_step_unit = os.environ.get('VALIDATION_STEP_UNIT')
model_variant = os.environ.get('MODEL_VARIANT')
dropout_rate = float(os.environ.get('DROPOUT_RATE'))
adata_file = os.environ.get('ADATA_FILE')
genome_ref = os.environ.get('GENOME_REF')
batch_subset = os.environ.get('BATCH_SUBSET').split(',')
locus_bed_file = os.environ.get('LOCUS_BED_FILE')

print(f"checking bfloat16 support...")

if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
	dtype = torch.bfloat16
	print(f"bfloat16 supported")
else:
	dtype = torch.float16
	print(f"bfloat16 not supported, switching to float16")

dpar = False
device = "cpu"

if torch.cuda.is_available():
	device = "cuda:0"
	if torch.cuda.device_count() > 1:
		print(f"detected {torch.cuda.device_count()} torch devices, using DataParallel")
		dpar=True
	else:
		print(f"detected {torch.cuda.device_count()} torch devices, not using DataParallel")

run_prefix = "_".join([
	f"time.{save_model_time}",
	f"neighbors.{n_neighbors_sc}",
	f"window.{window_length}",
	f"convlayers.{conv_layers}",
	f"ctrl_layers.{ctrl_layers}",
	f"schedMetric.{sched_metric}",
	f"jitter.{max_jitter}",
	f"trim.{trimming}",
	f"variant.{model_variant}",
])

save_dir = f"{run_data_dir}/{run_prefix}"
sc.settings.figdir = pathlib.Path(f"{save_dir}/figures")
# %%

adata = sc.read_h5ad(adata_file)

# Assuming adata.obs['category'] is a pandas Series
categories = np.unique(adata.obs['category'])

# Create a mapping dictionary
category_mapping = {category: idx for idx, category in enumerate(categories)}


subset_chroms = subset_chroms_env.split(',')
test_chroms = test_chroms_env.split(',')

train_chroms = [chrom for chrom in subset_chroms if chrom not in test_chroms]

bcmaps = collections.defaultdict(dict)
for ds in batch_subset:
	bcmaps[ds] = get_barcode_map(os.path.join(bc_files, ds, f"barcodes.txt"))

mapped_bcs = collections.defaultdict(list)
for ds in batch_subset:
	mapped_bcs[ds] = bcmaps[ds].values()

signals = collections.defaultdict(list)
for ds in batch_subset:
	signals[ds] = load_signals(os.path.join(signals_dir, ds, 'unstranded_directionality_merge_5000bp'), subset_bcs=list(mapped_bcs[ds]), chroms=subset_chroms)

combined_signals = {'pos': {}, 'neg': {}}
for signal_type in ['pos', 'neg']:
	# Get the list of chromosomes from the first dataset
	chromosomes = signals[next(iter(signals))][signal_type].keys()
	
	# Iterate over each chromosome
	for chr_key in chromosomes:
		# Collect matrices to be concatenated
		matrices_to_stack = [signals[dataset][signal_type][chr_key] for dataset in signals.keys()]
		
		# Concatenate the matrices using vstack
		combined_matrix = scipy.sparse.vstack(matrices_to_stack)
		
		# Store the combined matrix in the new dictionary
		combined_signals[signal_type][chr_key] = combined_matrix

# Combine cell labels
combined_barcodes = []
for dataset in signals.keys():
	combined_barcodes.extend([i + '-' + dataset for i in list(bcmaps[dataset].keys())])

# filter the adata object to only include the given barcodes
adata = adata[combined_barcodes]


# %%
print(f"extracting fasta sequences...")
sequences_faidx = pyfaidx.Fasta(genome_ref)
mapped_chroms = list(sequences_faidx.keys())
sequences = extract_fasta(filename=genome_ref, chroms=subset_chroms)
print(f"sequence extraction complete...")

config = {
	"use_cells": use_cells,
	"bias_features": bias_features,
	"bias_feature_embedding": bias_feature_embedding,
	"genome_ref": genome_ref,
	"adata_file": adata_file,
	"locus_bed_file": locus_bed_file,
	"n_neighbors_sc": n_neighbors_sc,
	"ctrl_nodes": ctrl_nodes,
	"ctrl_outputs": ctrl_outputs,
	"ctrl_layers": ctrl_layers,
	"bp_n_filters": bp_filters,
	"window": window_length,
	"counts_alpha": counts_alpha,
	"conv_layers": conv_layers,
	"trimming": trimming,
	"max_jitter": max_jitter,
	"category_mapping": category_mapping,
	"chrom_sizes": dict([(k,v.shape[1]) for k,v in sequences.items()]),
	"rseed": rseed,
	"lr": lr,
	"debug_strand": debug_strand,
	"sched_metric": sched_metric,
	"sched_patience": sched_patience,
	"sched_factor": sched_factor,
	"sched_thresh": sched_thresh,
	"dtype": dtype,
	"dpar": dpar,
	"train_batch_size": train_batch_size,
	"val_batch_size": val_batch_size,
	"cell_state_dim": 50,
	"dropout_rate": dropout_rate,
	"max_epochs": max_epochs,
	"validation_freq": validation_freq,
	"subset_chroms": subset_chroms,
	"train_chroms": train_chroms,
	"test_chroms": test_chroms,
	"batch_subset": batch_subset,
 	"bc_file": bc_files,
	"signals_dir": signals_dir,
	"validation_step_unit": validation_step_unit,
	"model_variant": model_variant
}
config['save_dir'] = save_dir
config['model_ckpts'] = f"{save_dir}/ckpt_model"
config['opt_ckpts'] = f"{save_dir}/ckpt_opt"

if os.environ.get('SLURM_JOB_ID') is not None:
	scontrol_result = subprocess.run(['scontrol', 'show', 'job', os.environ.get('SLURM_JOB_ID')], capture_output=True, text=True)
	stdout_path = None
	stderr_path = None

	# Extract the StdOut path from the output using grep
	if scontrol_result.returncode == 0:
		grep_result = subprocess.run(['grep', '-oP', 'StdOut=.*?(\S+)'], input=scontrol_result.stdout, capture_output=True, text=True)
		if grep_result.returncode == 0:
			stdout_path = grep_result.stdout.strip().split('=')[1]
			config['slurm_stdout_path'] = stdout_path
		grep_result = subprocess.run(['grep', '-oP', 'StdErr=.*?(\S+)'], input=scontrol_result.stdout, capture_output=True, text=True)
		if grep_result.returncode == 0:
			stderr_path = grep_result.stdout.strip().split('=')[1]
			config['slurm_stderr_path'] = stderr_path

if not os.path.exists(config['save_dir']):
	os.makedirs(config['save_dir'])
else:
	print(f"Save directory {config['save_dir']} already exists. Continuing.")

if not os.path.exists(config['model_ckpts']):
	os.makedirs(config['model_ckpts'])
else:
	print(f"Model ckpt directory {config['model_ckpts']} already exists. Continuing.")

if not os.path.exists(config['opt_ckpts']):
	os.makedirs(config['opt_ckpts'])
else:
	print(f"Opt ckpt directory {config['opt_ckpts']} already exists. Continuing.")


# %%
read_depths = get_read_depths(combined_signals)

# %%



sc.pp.neighbors(adata) # compute neighbors
dist_cells = adata.obsp['distances']
all_sorted_dists, all_sorted_inds, k_sorted_dists, k_sorted_inds = get_neighbors(dist_cells,config['n_neighbors_sc'])
config['neighbors'] = np.concatenate([arr.reshape(1, -1) for arr in k_sorted_inds],axis=0)


# get the loci dataframe from the bedfile
column_names = ['chrom', 'start', 'end', 'name', 'directionality', 'strand', None,'midpoint',None, None, None, None]
loci = pd.read_csv(locus_bed_file, compression='gzip', header=None, sep='\t')
loci.columns = column_names

config['random_state'] = np.random.RandomState(config['rseed'])


# get the log2 neighborhood read depth for each cell
read_depths['pos'] = read_depths['pos'][config['neighbors']].sum(axis=1)
read_depths['pos'] = np.log2(read_depths['pos'] + 1)
read_depths['pos'] = read_depths['pos'].reshape(-1, 1)

read_depths['neg'] = read_depths['neg'][config['neighbors']].sum(axis=1)
read_depths['neg'] = np.log2(read_depths['neg'] + 1)
read_depths['neg'] = read_depths['neg'].reshape(-1, 1)



if config['bias_features'] == 'all_genes' and config['bias_feature_embedding'] == 'pca':
	smoothed_cell_states = get_smoothed_cell_states(k_sorted_inds, adata.obsm['X_pca'])
	config['cell_states'] = (smoothed_cell_states - smoothed_cell_states.mean(axis=0, keepdims=True)) / smoothed_cell_states.std(axis=0, keepdims=True)
elif config['bias_features'] == 'tf' and config['bias_feature_embedding'] == 'pca':
	smoothed_cell_states = get_smoothed_cell_states(k_sorted_inds, adata.obsm['tf_pca'])
	config['cell_states'] = (smoothed_cell_states - smoothed_cell_states.mean(axis=0, keepdims=True)) / smoothed_cell_states.std(axis=0, keepdims=True)
	
# only use the centroids

def delete_rows_csc(mat, indices):
	coo_mat = mat.tocoo()
	coo_rows = coo_mat.row
	coo_col = coo_mat.col
	data = coo_mat.data
	keep_row_ind = ~np.isin(coo_rows, indices)
	mat_keep = coo_matrix((data[keep_row_ind], (coo_rows[keep_row_ind], coo_col[keep_row_ind])), shape=coo_mat.shape)
	return mat_keep

if config['use_cells'] != 'all':
	centroid_indices = adata.obs[config['use_cells']][adata.obs[config['use_cells']] == True].index.tolist()
	centroid_positions = np.where(adata.obs[config['use_cells']] == True)
	config["cell_labels"] = np.array(adata[adata.obs[config['use_cells']] == True].obs['category'].map(category_mapping))

	config['neighbors'] = config['neighbors'][centroid_positions]
	filtered_indices = [v for i,v in enumerate(k_sorted_inds) if i in centroid_positions[0]]
	unique_inds = np.unique([item for sublist in filtered_indices for item in sublist])
	delete_inds = np.setdiff1d(np.arange(0,adata.shape[0]),unique_inds)
 
	config['cell_states'] = config['cell_states'][centroid_positions]

	for signal_type in ['pos', 'neg']:
		# Get the list of chromosomes from the first dataset
		chromosomes = combined_signals[signal_type].keys()
		
		# Iterate over each chromosome
		for chr_key in chromosomes:
			print(f"zeroing non-neighbor cell signals for {chr_key}:{signal_type}, {combined_signals[signal_type][chr_key].shape}: nnz = {combined_signals[signal_type][chr_key].count_nonzero()}")
   
			matrix_coo = delete_rows_csc(combined_signals[signal_type][chr_key], delete_inds)

			combined_signals[signal_type][chr_key] = matrix_coo.tocsc()
			print(f"completed zeroing non-neighbor cell signals for {chr_key}:{signal_type}, {combined_signals[signal_type][chr_key].shape}: nnz = {combined_signals[signal_type][chr_key].count_nonzero()}")
			del matrix_coo

elif config['use_cells'] == 'all':
	config["cell_labels"] = np.array(adata.obs['category'].map(category_mapping))

train_dataset = MappedMultitaskDataset(
	sequence=sequences,
	signal_pos=combined_signals['pos'],
	signal_neg=combined_signals['neg'],
	loci=loci,
	neighbors=config['neighbors'],
	cell_states=config['cell_states'],
	cell_labels=np.array(adata.obs['category'].map(category_mapping)),
	read_depths_pos=read_depths['pos'],
	read_depths_neg=read_depths['neg'],
	trimming=config['trimming'], 
	window=config['window'],
	chroms=config['train_chroms'],
	random_state=0,
	max_jitter=config['max_jitter'],
	compute_locus_pbulk=False,
)

val_dataset = MappedMultitaskGenerator(
	sequence=sequences,
	signal_pos=combined_signals['pos'],
	signal_neg=combined_signals['neg'],
	loci=loci,
	neighbors=config['neighbors'],
	cell_states=config['cell_states'],
	cell_labels=np.array(adata.obs['category'].map(category_mapping)),
	read_depths_pos=read_depths['pos'],
	read_depths_neg=read_depths['neg'],
	trimming=config['trimming'], 
	window=config['window'],
	chroms=config['test_chroms'],
	random_state=0,
	max_jitter=config['max_jitter'],
	compute_locus_pbulk=True,
)


print(f"datasets initialized, deleting signals and sequences...")
del sequences, combined_signals
del signals

# Assuming you have a dataset called 'my_dataset' and 'split_ratio' represents the percentage of data for training


def window_filter(batch):
	# Filter the batch based on the window size
	filtered_batch = [x for x in batch if x[0].shape[1] == config['window']]
	
	# Unzip the filtered batch into individual components
	X_batch, y_batch, c_batch, r_batch, l_batch, start_batch, end_batch, chrom_batch, locus_idx_batch = zip(*filtered_batch)
	
	# Convert tensor components to tensors
	X_batch = torch.stack(X_batch)
	y_batch = torch.stack(y_batch)
	c_batch = torch.stack(c_batch)
	r_batch = torch.stack(r_batch)
	l_batch = torch.tensor(l_batch)
	start_batch = torch.tensor(start_batch)
	end_batch = torch.tensor(end_batch)
	locus_idx_batch = torch.tensor(locus_idx_batch)
	
	return (
		X_batch,
		y_batch,
		c_batch,
		r_batch,
		l_batch,
		start_batch,
		end_batch,
		list(chrom_batch),  # Keep as list of strings
		locus_idx_batch
	)

def window_filter_pbulk(batch):
	# Filter the batch based on the window size
	filtered_batch = [x for x in batch if x[0].shape[1] == config['window']]
	
	# Unzip the filtered batch into individual components
	X_batch, y_batch, y_pbulk_batch, c_batch, r_batch, l_batch, start_batch, end_batch, chrom_batch, locus_idx_batch = zip(*filtered_batch)
	
	# Convert tensor components to tensors
	X_batch = torch.stack(X_batch)
	y_batch = torch.stack(y_batch)
	y_pbulk_batch = torch.stack(y_pbulk_batch)
	c_batch = torch.stack(c_batch)
	r_batch = torch.stack(r_batch)
	l_batch = torch.tensor(l_batch)
	start_batch = torch.tensor(start_batch)
	end_batch = torch.tensor(end_batch)
	locus_idx_batch = torch.tensor(locus_idx_batch)
	
	return (
		X_batch,
		y_batch,
		y_pbulk_batch,
		c_batch,
		r_batch,
		l_batch,
		start_batch,
		end_batch,
		list(chrom_batch),  # Keep as list of strings
		locus_idx_batch
	)

train_loader = DataLoader(
	train_dataset, 
	batch_size=config['train_batch_size'], 
	shuffle=True, 
	pin_memory=True,
	collate_fn=window_filter,
	num_workers=num_workers)
val_loader = DataLoader(
	val_dataset, 
	batch_size=config['val_batch_size'], 
	shuffle=False, 
	pin_memory=True,
	collate_fn=window_filter_pbulk,
	num_workers=num_workers)

controller = CellStateController(
		n_inputs=config['cell_state_dim'], 
		n_nodes=config['ctrl_nodes'], 
		n_layers=config['ctrl_layers'], 
		n_outputs=config['ctrl_outputs'],
 	)

if config['model_variant']=="base":
	accessibility_model = DynamicBPNet(
			n_filters=config['bp_n_filters'], 
			n_layers=config['conv_layers'], 
			trimming=config['trimming'], 
			controller=controller,
			n_outputs=2,
		)
elif config['model_variant']=="regularized":
	accessibility_model = RegularizedDynamicBPNet(
			controller=controller,
			n_filters=config['bp_n_filters'], 
			n_layers=config['conv_layers'], 
			trimming=config['trimming'], 
			dropout_rate=config['dropout_rate'], 
			n_outputs=2,
		)
elif config['model_variant']=="regularized_counts":
	accessibility_model = RegularizedDynamicBPNetCounts(
			controller=controller,
			n_filters=config['bp_n_filters'], 
			n_layers=config['conv_layers'], 
			trimming=config['trimming'], 
			dropout_rate=config['dropout_rate'], 
			n_outputs=2,
		)



model = scBPnetCounts(accessibility_model, name=None)

if dpar:
	model = torch.nn.DataParallel(model)

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=config['sched_patience'], factor=config['sched_factor'],threshold=config['sched_thresh'])


wandb.init(
	# set the wandb project where this run will be logged
	project=wandb_proj_name,
	resume='allow',
	# track hyperparameters and run metadata
	config=config
)

config['wandb_run_id'] = wandb.run.id
config['wandb_run_url'] = wandb.run.get_url()

with open(f"{save_dir}/run_config.pkl", 'wb') as f:
	pickle.dump(config, f)

autocast=False
if torch.is_autocast_enabled():
	print(f"autocast detected, auto-casting model to {dtype}")
	autocast=True

print(f"starting training...")


for epoch in range(config['max_epochs']):

	running_loss = 0.0
	running_prof_loss = 0.0
	running_count_loss = 0.0
	train_batch_count = 0

	for batch_idx, batch in enumerate(tqdm(train_loader, leave=True, ncols=80, position=0)):
		model.train()
		X = batch[0].float().to(device)
		y = batch[1].float().to(device)
		c = batch[2].to(device)
		r = batch[3].to(device)
		optimizer.zero_grad()
		if autocast:
			with torch.autocast(device_type='cuda', dtype=dtype):
				y_profile, y_counts, = model(X, c)
		else:
			y_profile, y_counts = model(X, c)

		y_hat_ = torch.nn.functional.log_softmax(y_profile[:,config['debug_strand'],:],dim=-1)
		profile_loss = MNLLLoss(y_hat_, y[:,config['debug_strand'],:]).mean()
		count_loss = log1pMSELoss(y_counts.reshape(y_counts.shape[0], 1).squeeze(), y[:,config['debug_strand'],:].sum(dim=1).reshape(-1, 1).squeeze())

		profile_loss_ = profile_loss.item()
		count_loss_ = count_loss.item()

		running_prof_loss += profile_loss_
		running_count_loss += count_loss_
		running_loss += profile_loss_ + config['counts_alpha'] * count_loss_

		loss = profile_loss + config['counts_alpha'] * count_loss
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
		optimizer.step()

		running_loss += loss.item()

		train_batch_count += 1

		if batch_idx % config['validation_freq'] == 0 and config['validation_step_unit']=="batch":
			running_val_loss = 0.0
			running_val_profile_loss = 0.0
			running_val_count_loss = 0.0
			val_batch_count = 0

			with torch.no_grad():
				model.eval()
				for val_batch in val_loader:
					X_valid = val_batch[0].float().to(device)
					y_valid = val_batch[1].float().to(device)
					y_pbulk_valid = val_batch[2].float().to(device)
					c_valid = val_batch[3].to(device)
					r_valid = val_batch[4].to(device)
					if autocast:
						with torch.autocast(device_type='cuda', dtype=dtype):
							y_profile, y_counts, = model(X_valid, c_valid)
					else:
						y_profile, y_counts = model(X_valid, c_valid)

					y_hat_ = torch.nn.functional.log_softmax(y_profile[:,config['debug_strand'],:],dim=-1)
					profile_loss_valid = MNLLLoss(y_hat_, y_valid[:,config['debug_strand'],:]).mean()
					count_loss_valid = log1pMSELoss(y_counts.reshape(y_counts.shape[0], 1).squeeze(), y_valid[:,config['debug_strand'],:].sum(dim=1).reshape(-1, 1).squeeze())

					profile_loss_valid_ = profile_loss_valid.item()
					count_loss_valid_ = count_loss_valid.item()

					loss_valid = profile_loss_valid_ + config['counts_alpha'] * count_loss_valid_

					running_val_profile_loss += profile_loss_valid_
					running_val_count_loss += count_loss_valid_
					running_val_loss += profile_loss_valid_ + config['counts_alpha'] * count_loss_valid_

					y_profile = y_hat_.unsqueeze(1).unsqueeze(3)
					y_counts = y_counts.unsqueeze(1)

					y_valid = y_valid[:,config['debug_strand'],:]
					y_valid = y_valid.unsqueeze(1).unsqueeze(3)
					y_valid_counts = y_valid.sum(dim=2)

					measures = compute_performance_metrics(y_valid.cpu().numpy(), y_profile.cpu().numpy(),y_valid_counts.cpu().numpy(), y_counts.cpu().numpy(), 7, 81)

					y_pbulk_valid = torch.nn.functional.log_softmax(y_pbulk_valid[:,config['debug_strand'],:],dim=-1)
					y_pbulk_valid = y_pbulk_valid.unsqueeze(1).unsqueeze(3)
					y_pbulk_valid_counts = y_pbulk_valid.sum(dim=2)

					measures_pbulk = compute_performance_metrics(y_valid.cpu().numpy(), y_pbulk_valid.cpu().numpy(),y_valid_counts.cpu().numpy(), y_pbulk_valid_counts.cpu().numpy(), 7, 81)
					val_batch_count += 1

			print(f"epoch {epoch}, batch {batch_idx} writing metrics to wandb...")
			metrics_dict = {
						"loss": (running_loss / train_batch_count),
						"profile_loss": (running_prof_loss / train_batch_count),
						"count_loss": (running_count_loss / train_batch_count),
						"valid_loss": (running_val_loss / val_batch_count),
						"valid_profile_loss": (running_val_profile_loss / val_batch_count),
						"valid_count_loss": (running_val_count_loss / val_batch_count),
						"jsd":np.nan_to_num(measures['jsd'],nan=1).mean(),
						"nll": measures['nll'].mean(),
						"profile_pearson": np.nan_to_num(measures['profile_pearson'],nan=1).mean(),
						"profile_spearman": np.nan_to_num(measures['profile_spearman'],nan=1).mean(),
						"profile_mse": measures['profile_mse'].mean(),
						"count_pearson": measures['count_pearson'].mean(),
						"count_spearman": measures['count_spearman'].mean(),
						"count_mse": measures['count_mse'].mean(),
						"jsd_pbulk":np.nan_to_num(measures_pbulk['jsd'],nan=1).mean(),
						"nll_pbulk": measures_pbulk['nll'].mean(),
						"profile_pearson_pbulk": np.nan_to_num(measures_pbulk['profile_pearson'],nan=1).mean(),
						"profile_spearman_pbulk": np.nan_to_num(measures_pbulk['profile_spearman'],nan=1).mean(),
						"profile_mse_pbulk": measures_pbulk['profile_mse'].mean(),
						"count_pearson_pbulk": measures_pbulk['count_pearson'].mean(),
						"count_spearman_pbulk": measures_pbulk['count_spearman'].mean(),
						"count_mse_pbulk": measures_pbulk['count_mse'].mean(),
						"lr": optimizer.param_groups[0]['lr'],
	  		}
			wandb.log(metrics_dict)

			print(f"epoch {epoch}, batch {batch_idx} saving checkpoint to {config['save_dir']}")
			torch.save(model.state_dict(), f"{config['model_ckpts']}/model_e{epoch}_b{train_batch_count}.pth")
			# Save the optimizer state
			torch.save(optimizer.state_dict(), f"{config['opt_ckpts']}/optimizer_e{epoch}_b{train_batch_count}.pth")

			if config['sched_metric']=="train":
				scheduler.step(running_loss / train_batch_count)
			elif config['sched_metric']=="val":
				scheduler.step(running_val_loss / val_batch_count)


print("Finished Training")
wandb.finish()


