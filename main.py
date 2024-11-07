import yaml
from utils import ordered_yaml

import random
import torch
import wandb
import random


from trainers import (
    GNNTrainer,
    GNNCoraTrainer,
    GNNRedditTrainer,
    GNNPPITrainer,
    GNNCiteseerTrainer,
    GNNActorTrainer,
    BaselinesTrainer,
    GNNOGBTrainer
)

# Set seed
seed = 612
random.seed(seed)
torch.manual_seed(seed)

#############################################################
# Set modes:
# train: initialize trainer for classification
# eval: Evaluate the trained model quantitatively
#############################################################
mode = "train"


def main():
    config_file = "GAT_MIMIC3_readm.yml"
    config_path = f"./configs/{config_file}"

    with open(config_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {config_path}")


    if mode == "train":
        if config["train_type"] == "gnn":
            trainer = GNNTrainer(config)
        elif config["train_type"] == "gnncora":
            trainer = GNNCoraTrainer(config)
        elif config["train_type"] == "gnnreddit":
            trainer = GNNRedditTrainer(config)
        elif config["train_type"] == "gnnppi":
            trainer = GNNPPITrainer(config)
        elif config["train_type"] == "gnnciteseer":
            trainer = GNNCiteseerTrainer(config)
        elif config["train_type"] == "gnnactor":
            trainer = GNNActorTrainer(config)
        elif config["train_type"] == "gnnogb":
            trainer = GNNOGBTrainer(config)
        elif config["train_type"] == "baseline":
            trainer = BaselinesTrainer(config)
        else:
            raise NotImplementedError("This type of model is not implemented")
        trainer.train()


if __name__ == "__main__":
    main()
