import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import rank_zero_info
from tqdm import tqdm

from src.data.dataset import Dataset
from src.data.evaluator import Evaluator
from src.models.common.gluonts_predictor import GluonTSPredictor
from src.utils.enums import RunMode


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set seed and logging
    seed_everything(**cfg.seed)
    logging.basicConfig(**cfg.logging)
    logging.debug(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Determine dataset(s) to evaluate based on run mode
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) % len(cfg.data)
    sbatch_mode = cfg.run_mode == RunMode.SBATCH
    start_idx = task_id if sbatch_mode else cfg.start_idx
    end_idx = task_id + 1 if sbatch_mode else cfg.end_idx
    datasets = cfg.data[start_idx:end_idx]

    kwargs = {
        "desc": "Running evaluation",
        "total": len(datasets),
        "unit": "dataset",
        "disable": len(datasets) == 1,
    }

    for data in tqdm(datasets, **kwargs):        
        rank_zero_info(f"\n{'-' * 160}")
        
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
        predictor = GluonTSPredictor(
            forecaster=forecaster,
            imputation_method=instantiate(cfg.imputation),
            batch_size=cfg.data_batch_size,
        )

        # Load dataset
        dataset = Dataset(data.name, data.term)

        # Evaluate ensemble and save results
        evaluator = Evaluator(dataset, skip_existing=cfg.skip_existing)
        evaluator.evaluate(predictor)

        # Print ensemble weights for each cross-validation window
        forecaster.print_weights()

if __name__ == "__main__":
    main()
