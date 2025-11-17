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
    seed_everything(seed=cfg.seed, workers=cfg.workers, verbose=cfg.verbose)
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
        batch_size=cfg.data_batch_size,
        n_windows=cfg.n_windows,
        verbose=cfg.verbose,
    )

    # Prepare ensemble for evaluation
    predictor = GluonTSPredictor(forecaster)

    # Load list of dataset cfgs and use SLURM_ARRAY_TASK_ID to index the list.
    # Defaults to 22, which is the Ett1 daily dataset (short-term).
    cfg.data = cfg.data[int(os.environ.get("SLURM_ARRAY_TASK_ID", 22))]
    dataset_name, term = cfg.data.name, cfg.data.term    
    logging.info(f"Loading dataset: {dataset_name} ({term}-term)")
    dataset = Dataset(name=dataset_name, term=term)

    # Evaluate the ensemble and save the results
    evaluator = Evaluator(dataset=dataset, verbose=cfg.verbose)
    evaluator.evaluate(predictor=predictor)

    # Display the ensemble weights across all cross-validation windows
    forecaster.print_weights()


if __name__ == "__main__":
    main()
