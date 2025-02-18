import copy
from typing import Any
import os 
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch import nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from collections import Counter
from collections import defaultdict
import shutil 


KMER_LENGTH = 5

file_dir = os.path.dirname(__file__)



train_utils_path = os.path.join(file_dir,'..','train_utils')
training_data_path =  os.path.join(train_utils_path, 'train_data')
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
    
)

from utils_data import load_TF_data, SequenceDataset




class TrainLoop:
    def __init__(
        self,
        config: dict[str, Any],
        data: dict[str, Any],        
        model: torch.nn.Module,
        accelerator: Accelerator,
        epochs: int = 10000,
        log_step_show: int = 50,
        sample_epoch: int = 500,
        save_epoch: int = 500,
        model_name: str = "model_48k_sequences_per_group_K562_hESCT0_HepG2_GM12878_12k",
        num_sampling_to_compare_cells: int = 1000,
        run_name = ''
    ):
        # self.encode_data = data
        self.encode_data = data
        self.model = model
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)
        self.accelerator = accelerator
        self.epochs = epochs
        self.log_step_show = log_step_show
        self.sample_epoch = sample_epoch
        self.save_epoch = save_epoch
        self.model_name = model_name
        self.num_sampling_to_compare_cells = num_sampling_to_compare_cells
        self.batch_size = config.batch_size
        self.config = config
        
        if self.accelerator.is_main_process:
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

        # Metrics
        self.train_js, self.test_js, self.shuffle_js = 1, 1, 1
        self.seq_similarity = 1
        self.min_edit_distance, self.knn_distance, self.distance_from_closest = 100, 0, 0
        # for plots
        self.loss_values = []
        self.all_train_js, self.all_test_js, self.all_shuffle_js = [],[],[]
        self.all_seq_similarity = []
        self.test_js_by_cell_type_global_min = 1
        self.start_epoch = 1
        # count cell types in train
        # cell_dict_temp = Counter(self.encode_data['x_train_cell_type'].tolist())
        # # reoder by cell_types list
        # cell_dict = {k:cell_dict_temp[k] for k in self.encode_data['cell_types']}
        # # take only counts
        # cell_type_counts = list(cell_dict.values())
        # self.cell_type_probabilities = [x / sum(cell_type_counts) for x in cell_type_counts]


        self.cell_num_list = self.encode_data['cell_types']
        cell_list = list(self.encode_data["numeric_to_tag"].values())

        # count cell types in train
        cell_dict_temp = Counter(data['x_train_cell_type'].tolist())
        # reoder by cell_types list
        cell_dict = {k:cell_dict_temp[k] for k in self.cell_num_list}
        # take only counts
        cell_type_counts = list(cell_dict.values())
        self.cell_type_probabilities = [x / sum(cell_type_counts) for x in cell_type_counts]
        cell_num_sample_counts_list = [x * self.config.num_sampling_to_compare_cells for x in self.cell_type_probabilities]
        self.cell_num_sample_counts = {x:y for x,y in zip(self.cell_num_list, cell_num_sample_counts_list)}
        self.data_motif_cell_specific = {cell_type : {"train_motifs":self.encode_data["train_motifs_cell_specific"][cell_type], 
                                                "test_motifs":self.encode_data["test_motifs_cell_specific"][cell_type],
                                                "shuffle_motifs":self.encode_data["shuffle_motifs_cell_specific"][cell_type],
                                                }  for cell_type in cell_list}
        # Dataloader
        seq_dataset = SequenceDataset(seqs=self.encode_data["X_train"], c=self.encode_data["x_train_cell_type"], transform_ddsm = False)
        self.train_dl = DataLoader(seq_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

        self.log_dict = {}
        self.run_name = run_name
    
    def train_loop(self):
        # Prepare for training
        self.model, self.optimizer, self.train_dl = self.accelerator.prepare(self.model, self.optimizer, self.train_dl)
        
        for epoch in tqdm(range(self.start_epoch, self.epochs + 1)):
            # Getting loss of current batch
            for step, batch in enumerate(self.train_dl):
                loss = self.train_step(batch)
            # Sampling
            if epoch % self.sample_epoch == 0 and self.accelerator.is_main_process:
                #self.sample(epoch)
                self.sample_labelwise(epoch)
            # Logging loss
            if epoch % self.log_step_show == 0 and self.accelerator.is_main_process:
                self.log_step(loss, epoch)
            # Saving model
            if epoch % self.save_epoch == 0 and self.accelerator.is_main_process:
                self.save_model(epoch)
            
            self.loss_values.append(loss.mean().item())
        

    def train_step(self, batch):
        x,y = batch

        with self.accelerator.autocast():
            loss = self.model(x, y)

        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        self.accelerator.wait_for_everyone()
        self.optimizer.step()

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.ema.step_ema(self.ema_model, self.accelerator.unwrap_model(self.model))

        self.accelerator.wait_for_everyone()
        return loss

    def log_step(self, loss, epoch):
        if self.accelerator.is_main_process:
            # self.accelerator.log(
            #     {
            #         "train_js": self.train_js,
            #         "test_js": self.test_js,
            #         "shuffle_js": self.shuffle_js,
            #         "loss": loss.mean().item(),
            #         "seq_similarity": self.seq_similarity,
            #         "gc_ratio":self.gc_ratio,
            #         "edit_distance": self.min_edit_distance,
            #         "knn_distance":self.knn_distance,
            #         "distance_endogenous":self.distance_from_closest
            #     },
            #     step=epoch,
            # )
            self.log_dict.update({"loss": loss.mean().item()})
            self.accelerator.log(
                self.log_dict,
                step=epoch,
            )
    def save_model(self, epoch):
        print("saving")
        checkpoint_dict = {
            "model": self.accelerator.get_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "ema_model": self.accelerator.get_state_dict(self.ema_model),
        }
        torch.save(
            checkpoint_dict,
            f"checkpoints/epoch_{epoch}_{self.model_name}.pt",
        )

    def load(self, path):
        checkpoint_dict = torch.load(path)
        self.model.load_state_dict(checkpoint_dict["model"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
        self.start_epoch = checkpoint_dict["epoch"]

        if self.accelerator.is_main_process:
            self.ema_model.load_state_dict(checkpoint_dict["ema_model"])

        self.train_loop()
        # self.model, self.optimizer, self.train_dl = self.accelerator.prepare(self.model, self.optimizer, self.train_dl)
    
    def sample(self, epoch):
        self.model.eval()

        # Sample from the model
        print("sampling at epoch", epoch)
        synt_df, generated_sequences = create_sample(
            self.accelerator.unwrap_model(self.model),
            cell_types=self.encode_data["cell_types"],
            number_of_samples=self.num_sampling_to_compare_cells,
            sample_bs = self.config.sample_bs,
            cell_type_probabilities = self.cell_type_probabilities
        )
        self.seq_similarity = generate_similarity_using_train(self.encode_data["X_train"], self.config.seq_length)
        self.train_js = compare_motif_list(synt_df, self.encode_data["train_motifs"])
        self.test_js = compare_motif_list(synt_df, self.encode_data["test_motifs"])
        self.shuffle_js = compare_motif_list(synt_df, self.encode_data["shuffle_motifs"])
        print("Similarity", self.seq_similarity)
        print("js_TRAIN", self.train_js, "JS")
        print("js_TEST", self.test_js, "JS")
        print("js_SHUFFLE", self.shuffle_js, "JS")
        self.all_seq_similarity.append(self.seq_similarity)
        self.all_train_js.append(self.train_js)
        self.all_test_js.append(self.test_js)
        self.all_shuffle_js.append(self.shuffle_js)

        train_sequences = self.encode_data["train_sequences"]
        self.gc_ratio = gc_content_ratio(train_sequences, generated_sequences)
        self.min_edit_distance = min_edit_distance_between_sets(generated_sequences, train_sequences)
        train_vectors, generated_vectors = generate_kmer_frequencies(train_sequences, generated_sequences, KMER_LENGTH)
        self.knn_distance = knn_distance_between_sets(generated_vectors, train_vectors)
        self.distance_from_closest = distance_from_closest_in_second_set(generated_vectors, train_vectors)
        print("Min_Edit", self.min_edit_distance)
        print("KNN_distance", self.knn_distance)
        print("Distance_endogenous", self.distance_from_closest)
        self.model.train()

    def sample_labelwise(self, epoch):
        print("sampling at epoch", epoch)
        generated_motif, generated_celltype_motif, generated_celltype_sequences = self.create_sample_labelwise()
        generated_sequences = [sequence for sequences in generated_celltype_sequences.values() for sequence in sequences]
        train_sequences = self.encode_data["train_sequences"]
        # first calculate validation metrics in bulk
        seq_similarity = generate_similarity_using_train(self.encode_data["X_train"], self.config.seq_length)
        train_js, test_js, shuffle_js, gc_ratio, min_edit_distance, knn_distance, distance_from_closest = calculate_validation_metrics(self.encode_data, train_sequences, generated_motif, generated_sequences, get_kmer_metrics = True, kmer_length = self.config.kmer_length,)
        # train_js, test_js, shuffle_js, gc_ratio, min_edit_distance = calculate_validation_metrics(
        #     self.encode_data, train_sequences, generated_motif, generated_sequences, get_kmer_metrics = False, kmer_length = self.config.kmer_length,
        #     )
        self.log_dict.update({
            "train_js": train_js,
            "test_js": test_js,
            "shuffle_js": shuffle_js,
            "edit_distance": min_edit_distance,
            "knn_distance":knn_distance,
            "distance_endogenous":distance_from_closest,
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
        print("KNN_distance", knn_distance)
        print("Distance_endogenous", distance_from_closest)
        # now, generate metrics cell_type-wise and then average it
        validation_metric_cell_specific = defaultdict(list)
        for cell_num in self.cell_num_list:
            cell_type = self.encode_data['numeric_to_tag'][cell_num]
            new_data = self.data_motif_cell_specific[cell_type]
            train_sequences_cell_specific = self.encode_data["train_sequences_cell_specific"][cell_type]
            generated_motif_cell_specific = generated_celltype_motif[cell_type]
            generated_sequences_cell_specific = generated_celltype_sequences[cell_type]
            train_js, test_js, shuffle_js, gc_ratio, min_edit_distance, knn_distance, distance_from_closest = calculate_validation_metrics(new_data, train_sequences_cell_specific, generated_motif_cell_specific, generated_sequences_cell_specific, self.config.kmer_length)
            train_js, test_js, shuffle_js, gc_ratio, min_edit_distance = calculate_validation_metrics(
                new_data, train_sequences_cell_specific, generated_motif_cell_specific, generated_sequences_cell_specific, get_kmer_metrics=False, kmer_length= self.config.kmer_length)
            # adding to dict
            validation_metric_cell_specific["train_js_by_cell_type"].append(train_js)
            validation_metric_cell_specific["test_js_by_cell_type"].append(test_js)
            validation_metric_cell_specific["shuffle_js_by_cell_type"].append(shuffle_js)
            validation_metric_cell_specific["min_edit_distance_by_cell_type"].append(min_edit_distance)
            validation_metric_cell_specific["knn_distance_by_cell_type"].append(knn_distance)
            validation_metric_cell_specific["distance_from_closest_by_cell_type"].append(distance_from_closest)
            validation_metric_cell_specific["gc_ratio_by_cell_type"].append(gc_ratio)
            
        
        validation_metric_average = {x:np.mean(y) for x,y in validation_metric_cell_specific.items()}
        if validation_metric_average["test_js_by_cell_type"] < self.test_js_by_cell_type_global_min:
            fasta_file = f"{training_data_path}/{self.run_name}synthetic_motifs.fasta"
            bed_file = f"{training_data_path}/{self.run_name}syn_results_motifs.bed"
            # force renaming even if file exists
            shutil.move(fasta_file, rename_file_to_best(fasta_file))
            shutil.move(bed_file, rename_file_to_best(bed_file))
            
        # saving to logging dictionary
        self.log_dict.update(validation_metric_average)
        print("cell type-wise \n",validation_metric_average)  

    def create_sample_labelwise(self):
        # keeping label ratio from train data in generated sequences
        diffusion_model = self.accelerator.unwrap_model(self.model)
        # cond_weight_to_metric = 0 # no guidance
        cond_weight_to_metric = 1
        self.model.eval()
        nucleotides = ["A", "C", "G", "T"]
        generated_celltype_motif = {}
        generated_celltype_sequences = {}
        bulk_final_sequences = []
        for cell_num in self.cell_num_list:
            cell_type = self.encode_data['numeric_to_tag'][cell_num]
            final_sequences = []
            plain_generated_sequences = []
            cell_type_sample_count = self.cell_num_sample_counts[cell_num]
            if cell_type_sample_count < 100:
                cell_type_sample_count = 100
            cell_type_sample_size = int(cell_type_sample_count / self.config.sample_bs)
            print(f"Generating {int(cell_type_sample_count) // self.config.sample_bs * self.config.sample_bs} samples for cell_type {cell_type}")
            for n_a in range(cell_type_sample_size):
                sampled_cell_types = np.array([cell_num] * self.config.sample_bs)
                classes = torch.from_numpy(sampled_cell_types).float().to(diffusion_model.device)
                sampled_images = diffusion_model.sample(classes, (self.config.sample_bs, 1, 4, 200), cond_weight_to_metric)
                for n_b, image in enumerate(sampled_images[-1]):
                    sequence = "".join([nucleotides[s] for s in np.argmax(image.reshape(4, 200), axis=0)])
                    plain_generated_sequences.append(sequence)
                    seq_final = f">seq_test_{cell_num}_{n_a}_{n_b}\n" + sequence
                    final_sequences.append(seq_final)

            bulk_final_sequences += final_sequences
            # extract motifs from generated sequences
            df_motifs_count_syn = extract_motifs(final_sequences, self.run_name)
            generated_celltype_motif[cell_type] = df_motifs_count_syn
            generated_celltype_sequences[cell_type] = plain_generated_sequences
        generated_motif = extract_motifs(bulk_final_sequences, self.run_name)
        self.model.train()
        return generated_motif, generated_celltype_motif, generated_celltype_sequences



def create_sample(
    diffusion_model,
    cell_type_probabilities,
    cell_types: list,
    number_of_samples: int = 1000,
    cond_weight_to_metric: int = 0,
    sample_bs:int = 10
):
    nucleotides = ["A", "C", "G", "T"]
    final_sequences = []
    for n_a in range(int(number_of_samples/ sample_bs)):
        sampled = torch.from_numpy(np.random.choice(cell_types, sample_bs, p=cell_type_probabilities))
        classes = sampled.float().to(diffusion_model.device)
        sampled_images = diffusion_model.sample(classes, (sample_bs, 1, 4, 200), cond_weight_to_metric)
        plain_generated_sequences = []
        for n_b, x in enumerate(sampled_images[-1]):
            sequence = "".join([nucleotides[s] for s in np.argmax(x.reshape(4, 200), axis=0)])
            plain_generated_sequences.append(sequence)
            seq_final = f">seq_test_{n_a}_{n_b}\n" + sequence
            final_sequences.append(seq_final)
            
    df_motifs_count_syn = extract_motifs(final_sequences)
    return df_motifs_count_syn, plain_generated_sequences



class EMA:
    # https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py
    def __init__(self, beta: float = 0.995) -> None:
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model: nn.Module, current_model: nn.Module) -> None:
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        device = new.device
        old = old.to(device)
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model: nn.Module, model: nn.Module, step_start_ema: int = 500) -> None:
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

def rename_file_to_best(file_path):
    dir_name, base_name = os.path.split(file_path)
    file_name, file_ext = os.path.splitext(base_name)

    new_file_name = 'best_' + file_name + file_ext

    return os.path.join(dir_name, new_file_name)
