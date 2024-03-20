from typing import Optional

import argparse

import torch
import wandb

from captcha_competition import ConfigKeys, load_config, MODELS_PATH
from captcha_competition.training import trainer_factory


DEFAULT_CONFIG_FILENAME = "resnet-transformer-4.yaml"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_configuration_filename() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_filename",
        nargs="?",
        default=DEFAULT_CONFIG_FILENAME,
        help="Name of the configuration file",
    )
    return parser.parse_args().config_filename


def main(config_filename: Optional[str] = None):
    if config_filename is None:
        config_filename = get_configuration_filename()
    config = load_config(config_filename)

    wandb.init(project="captcha_competition_tuesday", config=config)

    # The name is the name of the configuration file without the extension
    model_name = config_filename.split(".")[0]
    if wandb.run is not None:
        wandb.run.name = model_name

    trainer = trainer_factory(
        model_params=config[ConfigKeys.MODEL],
        optimizer_params=config[ConfigKeys.OPTIMIZER],
        preprocessing_params=config[ConfigKeys.PREPROCESSING],
        train_dataset_params=config[ConfigKeys.TRAIN_DATASET],
        val_dataset_params=config[ConfigKeys.VAL_DATASET],
        dataloader_params=config[ConfigKeys.DATALOADER],
        trainer_params=config[ConfigKeys.TRAINER],
        model_name=model_name,
    )
    model = trainer.model
    pt_filename = MODELS_PATH / f"{config_filename.split('.')[0]}.pt"
    state = torch.load(pt_filename)
    model.load_state_dict(torch.load(pt_filename)["model_state_dict"])
    model = model.to(DEVICE)
    trainer.model = model

    trainer.best_eval_accuracy = state["val_accuracy"]
    optimizer = trainer.optimizer
    optimizer.load_state_dict(state["optimizer_state_dict"])

    trainer.train()


if __name__ == "__main__":
    main()
