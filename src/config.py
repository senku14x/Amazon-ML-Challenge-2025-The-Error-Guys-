"""
Configuration management for the pipeline.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class PathConfig:
    """File paths configuration."""
    base_dir: Path
    data_dir: Path
    output_dir: Path
    embeddings_dir: Path
    models_dir: Path
    
    @classmethod
    def from_base(cls, base_dir: str):
        base = Path(base_dir)
        return cls(
            base_dir=base,
            data_dir=base / "dataset",
            output_dir=base / "output",
            embeddings_dir=base / "embeddings",
            models_dir=base / "models"
        )


@dataclass
class ModelConfig:
    """Model hyperparameters."""
    seed: int = 42
    n_folds: int = 5
    
    # LightGBM
    lgb_params: dict = None
    
    # XGBoost
    xgb_params: dict = None
    
    # CatBoost
    cat_params: dict = None
    
    def __post_init__(self):
        if self.lgb_params is None:
            self.lgb_params = {
                'objective': 'regression_l1',
                'metric': 'mae',
                'learning_rate': 0.03,
                'num_leaves': 31,
                'min_data_in_leaf': 400,
                'feature_fraction': 0.55,
                'bagging_fraction': 0.7,
                'bagging_freq': 1,
                'lambda_l1': 8.0,
                'lambda_l2': 20.0,
                'verbosity': -1,
                'seed': self.seed
            }
        
        if self.xgb_params is None:
            self.xgb_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'learning_rate': 0.03,
                'max_depth': 7,
                'subsample': 0.7,
                'colsample_bytree': 0.55,
                'lambda': 20.0,
                'alpha': 8.0,
                'tree_method': 'hist',
                'seed': self.seed
            }
        
        if self.cat_params is None:
            self.cat_params = {
                'loss_function': 'MAE',
                'learning_rate': 0.03,
                'depth': 7,
                'l2_leaf_reg': 30.0,
                'bootstrap_type': 'Bernoulli',
                'subsample': 0.7,
                'iterations': 10000,
                'early_stopping_rounds': 100,
                'random_seed': self.seed,
                'verbose': 250
            }


@dataclass
class EmbeddingConfig:
    """Embedding generation configuration."""
    # Text
    e5_model: str = "intfloat/e5-large-v2"
    e5_max_length: int = 512
    e5_batch_size: int = 32
    
    # Image
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    clip_batch_size: int = 192
    
    # PCA
    pca_components: int = 128
    
    # Device
    device: str = "cuda"
