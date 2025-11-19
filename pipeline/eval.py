import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from src.data.dataset import Dataset
from src.data.evaluator import Evaluator
from src.models.common.gluonts_predictor import GluonTSPredictor


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set seed and logging
    seed_everything(**cfg.seed)
    logging.basicConfig(**cfg.logging)
    logging.debug(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Load models
    models = [
        instantiate(model_cfg, batch_size=cfg.model_batch_size)
        for model_cfg in cfg.models.values()
    ]

    # Create ensemble forecaster
    forecaster = instantiate(
        cfg.ensemble,
        models=models,
        metric=cfg.metric,
        n_windows=cfg.n_windows,
    )

    # Wrap ensemble forecaster with predictor for evaluation
    predictor = GluonTSPredictor(forecaster)

    # Use SLURM_ARRAY_TASK_ID to index list of dataset name-term pairs
    cfg.data = cfg.data[int(os.environ.get("SLURM_ARRAY_TASK_ID"))]
    dataset_name, term = cfg.data.name, cfg.data.term
    dataset = Dataset(dataset_name, term)

    # Evaluate ensemble and save results
    evaluator = Evaluator(dataset)
    evaluator.evaluate(predictor)

    # Print ensemble weights for each cross-validation window
    forecaster.print_weights()


if __name__ == "__main__":
    main()
