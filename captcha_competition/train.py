import argparse

import wandb

from captcha_competition import ConfigKeys, load_config
from captcha_competition.training import trainer_factory


DEFAULT_CONFIG_FILENAME = "efficientnet_default.yaml"


def get_configuration_filename() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_filename",
        nargs="?",
        default=DEFAULT_CONFIG_FILENAME,
        help="Name of the configuration file",
    )
    return parser.parse_args().config_filename


def main():
    config_filename = get_configuration_filename()
    config = load_config(config_filename)

    wandb.init(project="captcha_competition_debugging", config=config)
    trainer = trainer_factory(
        model_params=config[ConfigKeys.MODEL],
        optimizer_params=config[ConfigKeys.OPTIMIZER],
        preprocessing_params=config[ConfigKeys.PREPROCESSING],
        train_dataset_params=config[ConfigKeys.TRAIN_DATASET],
        val_dataset_params=config[ConfigKeys.VAL_DATASET],
        dataloader_params=config[ConfigKeys.DATALOADER],
        trainer_params=config[ConfigKeys.TRAINER],
    )
    trainer.train()


if __name__ == "__main__":
    main()
