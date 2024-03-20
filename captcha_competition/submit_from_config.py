from pathlib import Path

import torch
import pandas as pd
import tqdm  # type: ignore[import-untyped]

from captcha_competition import (
    ConfigKeys,
    load_config,
    MODELS_PATH,
    DATA_PATH,
    DATA_RAW_PATH,
)
from captcha_competition.training import (
    trainer_factory,
    dataset_factory,
    preprocessing_fc_factory,
    DataLoaderHandler,
    CustomAccuracyMetric,
)

SUBMISSION_FILENAME = "resnet-transformer-82-finetune.yaml"
MODEL_NAME = "resnet-transformer-8"
MAPPING = {
    10: "a",
    11: "e",
    12: "u",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_submission_file(config_filename: str):
    config = load_config(config_filename)

    model_params = config[ConfigKeys.MODEL]
    model_params["num_classes"] = 13
    trainer = trainer_factory(
        model_params=config[ConfigKeys.MODEL],
        optimizer_params=config[ConfigKeys.OPTIMIZER],
        preprocessing_params=config[ConfigKeys.PREPROCESSING],
        train_dataset_params=config[ConfigKeys.TRAIN_DATASET],
        val_dataset_params=config[ConfigKeys.VAL_DATASET],
        dataloader_params=config[ConfigKeys.DATALOADER],
        trainer_params=config[ConfigKeys.TRAINER],
        model_name=MODEL_NAME,
    )

    model = trainer.model

    pt_filename = MODELS_PATH / f"{MODEL_NAME}.pt"

    model.load_state_dict(torch.load(pt_filename)["model_state_dict"])
    model = model.to(DEVICE)
    # Create test dataset
    preprocessing_fc = preprocessing_fc_factory(
        model_type=trainer.model_type, **config[ConfigKeys.PREPROCESSING]
    )

    test_dataset = dataset_factory(
        preprocessing_fc=preprocessing_fc, **config[ConfigKeys.TEST_DATASET]
    )

    # Create test dataloader
    data_loader_config = config[ConfigKeys.DATALOADER]
    data_loader_config["shuffle"] = False
    data_loader_config["steps_per_epoch"] = None
    test_dataloader = DataLoaderHandler(test_dataset, **data_loader_config)

    # Make predictions
    all_predictions_list = []
    with torch.no_grad():
        for images, _ in tqdm.tqdm(test_dataloader):
            # print(label)
            outputs = model(images)
            predictions = CustomAccuracyMetric.get_predicted_classes(outputs)
            predictions_list = predictions.tolist()
            all_predictions_list.extend(predictions_list)
    all_predictions_list = [
        "".join(map(_to_char, pred)) for pred in all_predictions_list
    ]
    # Create submission file
    test_path = Path(test_dataloader.dataset.raw_img_dir)  # type: ignore[union-attr]
    images_ids = _get_images_ids(test_path)

    submission_df = pd.DataFrame(
        {"Id": images_ids, "Label": all_predictions_list}
    )
    submission_df.to_csv(
        DATA_PATH / f"{config_filename.split('.')[0]}.csv", index=False
    )


def _to_char(label: int) -> str:
    if label < 10:
        return str(label)

    return MAPPING[label]


def _get_images_ids(folder_path: Path) -> list[str]:
    return sorted(
        [int(image_path.stem) for image_path in folder_path.glob("*.png")]
    )


def create_test_csv():
    # Create a test.csv file in DATA_RAW_PATH with dummy labels (000000)
    ids = list(range(0, 2500))
    labels = ["000000"] * 2500
    test_df = pd.DataFrame({"Id": ids, "Label": labels})
    test_df.to_csv(DATA_RAW_PATH / "test.csv", index=False)


if __name__ == "__main__":
    create_submission_file(SUBMISSION_FILENAME)
    print("Submission file created")
    # create_test_csv()
