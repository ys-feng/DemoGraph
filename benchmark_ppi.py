import yaml
from utils import ordered_yaml

import pickle
import random
import torch
import wandb


from trainers import (
    GNNTrainer,
    GNNCoraTrainer,
    GNNRedditTrainer,
    GNNPPITrainer,
    GNNCiteseerTrainer,
    GNNActorTrainer,
    BaselinesTrainer
)

def benchmark_baselines(config):
    # Initialize baseline models
    with open(config["datasets"]["dataset_path"], 'rb') as inp:
        unp = pickle.Unpickler(inp)
        mimic3base = unp.load()

    for method in [
        # "DrAgent",
        # "StageNet",
        # "AdaCare",
        # "Transformer",
        # "RNN",
        # "ConCare",
        # "GRSAP",
        # "Deepr",
        # "MICRON",
        # "GAMENet",
        "MoleRec",
        # "SafeDrug",
        # "SparcNet",
    ]:
        for task in [
            "readm",
            # "mort_pred",
            # "los",
            # "drug_rec"
        ]:
            config["train"]["baseline_name"] = method
            config["train"]["task"] = task
            dataset_name = config["datasets"]["name"]
            config["checkpoint"]["path"] = f"./checkpoints/{method}/{dataset_name}/{task}/"
            print(f"Training {method} on task {task}")

            # Remember to update the trainer name
            trainer = BaselinesTrainer(config)
            trainer.train()
            del trainer


def benchmark_gnns(config):
    # Load GNN configs
    with open("./configs/GNN/GNN_Configs.yml", mode='r') as f:
        loader, _ = ordered_yaml()
        gnn_config = yaml.load(f, loader)

    for archi in [
        # "GCN",
        # "GAT",
        "GIN",
        "HetRGCN",
        # "HGT"
    ]:
        config["GNN"] = gnn_config[archi]
        dataset_name = config["datasets"]["name"]
        config["name"] = f"{archi}_MTCausal_MIMIC{dataset_name[-1]}_RMDL"
        config["checkpoint"]["path"] = f"./checkpoints/GNN_ablation/{dataset_name}/{archi}/"
        config["logging"]["tags"] += [archi]

        trainer = GNNTrainer(config)
        trainer.train()
        wandb.finish()
        del trainer


def benchmark_dropouts(config):
    for dp in [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6
    ]:
        config["GNN"]["feat_drop"] = dp
        config["name"] = f"HGT_MTCausal_MIMIC3_RMDL_dp{dp}"
        dataset_name = config["datasets"]["name"]
        config["checkpoint"]["path"] = f"./checkpoints/Dropout_ablation/{dataset_name}/{dp}/"
        config["logging"]["tags"] += ["abl_dropout"]

        trainer = GNNTrainer(config)
        trainer.train()
        wandb.finish()
        del trainer


def benchmark_hidden_dim(config):
    for dim in [
        16, 32, 64, 128, 256
    ]:
        config["GNN"]["hidden_dim"] = dim
        config["name"] = f"HGT_MTCausal_MIMIC3_RMDL_dim{dim}"
        dataset_name = config["datasets"]["name"]
        config["checkpoint"]["path"] = f"./checkpoints/Hidden_Dim_ablation/{dataset_name}/{dim}/"
        config["logging"]["tags"] += ["abl_dim"]

        trainer = GNNTrainer(config)
        trainer.train()
        wandb.finish()
        del trainer

def benchmark_lr(config):
    for lr in [
        0.01, 0.005, 0.001, 0.0005, 0.0001
    ]:
        config["optimizer"]["lr"] = lr
        config["name"] = f"Base_GAT_MIMIC3_{lr}"
        dataset_name = config["datasets"]["name"]
        config["checkpoint"]["path"] = f"./checkpoints/lr/{dataset_name}/{lr}/"
        config["logging"]["tags"] += ["abl_lr"]

        trainer = GNNTrainer(config)
        trainer.train()
        wandb.finish()
        del trainer


def benchmark_hyperparams(config):

# For GAT:

    # for dim in [
    #     256,
    #     128,
    # ]:
    #     for d in [
    #         0.05
    #         # 0.6, 0.4,
    #         # 0.2, 0.15,
    #         # 0.1, 0.05,
    #     ]:
    #         for l in [
    #             2,
    #             # 1
    #         ]:
    #             for lr in [
    #                 0.01, 0.0001,
    #                 # 0.05, 0.001, 0.005,
    #             ]:
    #                 for heads in [
    #                     16,
    #                     # 8,
    #                 ]:
    #                     for nec in [
    #                         500,
    #                         100
    #                     ]:
    #                         config["GNN"]["attn_drop"] = d
    #                         config["GNN"]["feat_drop"] = d
    #                         config["GNN"]["hidden_dim"] = dim
    #                         config["GNN"]["num_layers"] = l
    #                         config["GNN"]["num_heads"] = heads
    #                         config["optimizer"]["lr"] = lr
    #                         config["GNN"]["nec"] = nec
    #                         config["name"] = f"PPI_CGA_GAT_d{d}dim{dim}L{l}h{heads}lr{lr}nec{nec}"
    #                         dataset_name = config["datasets"]["name"]
    #                         config["checkpoint"]["path"] = f"./checkpoints/HParameters{dataset_name}_PPI_CGA_GAT/d{d}dim{dim}L{l}h{heads}lr{lr}nec{nec}/"
    #
    #                         config["logging"]["tags"] += ["hp"]
    #
    #                         trainer = GNNPPITrainer(config)
    #                         trainer.train()
    #                         wandb.finish()
    #                         del trainer

#For GraphSAGE:
    #
    # for dim in [
    #     256, 128, 64,
    #     # 32
    # ]:
    #     for d in [
    #         0.05, 0.1, 0.15, 0.2, 0.01
    #         # 0.3
    #     ]:
    #         for l in [
    #             # 4,
    #             3, 2, 1
    #         ]:
    #             for lr in [
    #                 # 0.00005, 0.0001, 0.0005, 0.001,
    #                 0.005, 0.01
    #             ]:
    #                 for nec in [
    #                     3, 30,
    #                     # 5, 10, 50
    #                 ]:
    #                     # config["GNN"]["attn_drop"] = d
    #                     config["GNN"]["feat_drop"] = d
    #                     config["GNN"]["hidden_dim"] = dim
    #                     config["GNN"]["num_layers"] = l
    #                     config["optimizer"]["lr"] = lr
    #                     config["GNN"]["nec"] = nec
    #                     config["name"] = f"PPI_CGA_SAGE_d{d}dim{dim}L{l}lr{lr}nec{nec}"
    #                     dataset_name = config["datasets"]["name"]
    #                     config["checkpoint"][
    #                         "path"] = f"./checkpoints/HParameters{dataset_name}_PPI_CGA_SAGE/d{d}dim{dim}L{l}lr{lr}nec{nec}/"
    #
    #                     config["logging"]["tags"] += ["hp"]
    #
    #                     trainer = GNNPPITrainer(config)
    #                     trainer.train()
    #                     wandb.finish()
    #                     del trainer

# For GCN:

    # for dim in [
    #     128, 256, 64,
    #     # 32
    # ]:
    #     for d in [
    #         0.15, 0.05, 0.1
    #         # 0.3
    #     ]:
    #         for l in [
    #             # 4,
    #             # 3,
    #             2, 1
    #         ]:
    #             for lr in [
    #                 # 0.00005, 0.0001, 0.0005, 0.001,
    #                 0.01, 0.001, 0.005,
    #             ]:
    #                 for nec in [
    #                     3, 100
    #                     # 5, 10, 50
    #                 ]:
    #                     # config["GNN"]["attn_drop"] = d
    #                     config["GNN"]["feat_drop"] = d
    #                     config["GNN"]["hidden_dim"] = dim
    #                     config["GNN"]["num_layers"] = l
    #                     config["optimizer"]["lr"] = lr
    #                     config["GNN"]["nec"] = nec
    #                     config["name"] = f"PPI_CGA_GCN_d{d}dim{dim}L{l}lr{lr}nec{nec}"
    #                     dataset_name = config["datasets"]["name"]
    #                     config["checkpoint"][
    #                         "path"] = f"./checkpoints/HParameters{dataset_name}_PPI_CGA_GCN/d{d}dim{dim}L{l}lr{lr}nec{nec}/"
    #                     config["logging"]["tags"] += ["hp"]
    #
    #                     trainer = GNNPPITrainer(config)
    #                     trainer.train()
    #                     wandb.finish()
    #                     del trainer



# For GIN:

    for dim in [
        256,
        128,
        # 64,
        # 32
    ]:
        for d in [
            # 0.6, 0.4,
            # 0.3, 0.2
            0.1, 0.15,
            # 0.05, 0.05,

        ]:
            for l in [
                # 4,
                2,
                # 1,
            ]:
                for lr in [
                    # 0.01, 0.001, 0.0001
                    0.05, 0.001, 0.005,
                ]:
                    for nec in [
                        100, 300
                        # 5, 10, 50
                    ]:
                        # config["GNN"]["attn_drop"] = d
                        config["GNN"]["feat_drop"] = d
                        config["GNN"]["hidden_dim"] = dim
                        config["GNN"]["num_layers"] = l
                        config["optimizer"]["lr"] = lr
                        config["GNN"]["nec"] = nec
                        config["name"] = f"PPI_CGA_GIN_d{d}dim{dim}L{l}lr{lr}nec{nec}dyn"
                        dataset_name = config["datasets"]["name"]
                        config["checkpoint"][
                            "path"] = f"./checkpoints/HParameters{dataset_name}_PPI_CGA_GIN/d{d}dim{dim}L{l}lr{lr}nec{nec}dyn/"
                        config["logging"]["tags"] += ["hp"]

                        trainer = GNNPPITrainer(config)
                        trainer.train()
                        wandb.finish()
                        del trainer

# Set seed
seed = 611
random.seed(seed)
torch.manual_seed(seed)

# config_file = "Baselines_MIMIC4.yml"
config_file = "GIN_PPI.yml"
config_path = f"./configs/{config_file}"

with open(config_path, mode='r') as f:
    loader, _ = ordered_yaml()
    config = yaml.load(f, loader)
    print(f"Loaded configs from {config_path}")

if __name__ == "__main__":
    # benchmark_baselines(config)
    # benchmark_gnns(config)
    # benchmark_dropouts(config)
    # benchmark_hidden_dim(config)
    # benchmark_lr(config)
    benchmark_hyperparams(config)