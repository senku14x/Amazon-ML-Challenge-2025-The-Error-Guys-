"""
Gradient boosting models and ensemble strategy.

Models:
- LightGBM (primary)
- XGBoost (diversity)
- CatBoost (regularization)

Ensemble:
- Optimal weighted blending via grid search
"""

from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from joblib import dump as joblib_dump


# ============================================================
# METRICS
# ============================================================

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute SMAPE (Symmetric Mean Absolute Percentage Error).
    
    Formula: 100 × mean(|pred - true| / ((|pred| + |true|) / 2))
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    out = np.zeros_like(denom)
    out[mask] = np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return float(out.mean() * 100.0)


def to_log(y: np.ndarray) -> np.ndarray:
    """Transform target to log-space."""
    return np.log1p(np.clip(y, 0, None))


def from_log(y_log: np.ndarray) -> np.ndarray:
    """Transform log-space predictions back to price."""
    return np.expm1(y_log)


# ============================================================
# LIGHTGBM MODEL
# ============================================================

def train_lightgbm_cv(
    X: np.ndarray,
    y_log: np.ndarray,
    X_test: np.ndarray,
    n_folds: int = 5,
    params: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train LightGBM with K-fold cross-validation.
    
    Returns:
        oof_predictions, test_predictions
    """
    print("\n" + "=" * 60)
    print("LIGHTGBM TRAINING (K-FOLD CV)")
    print("=" * 60)
    
    if params is None:
        params = {
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
            'seed': seed
        }
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros((len(X_test), n_folds))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")
        
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_log[train_idx], y_log[val_idx]
        
        ds_tr = lgb.Dataset(X_tr, label=y_tr)
        ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr)
        
        model = lgb.train(
            params,
            ds_tr,
            num_boost_round=5000,
            valid_sets=[ds_tr, ds_val],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(250)
            ]
        )
        
        # Predictions
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        test_preds[:, fold-1] = model.predict(X_test, num_iteration=model.best_iteration)
        
        # Score
        val_smape = smape(from_log(y_val), from_log(oof_preds[val_idx]))
        fold_scores.append(val_smape)
        print(f"Fold {fold} SMAPE: {val_smape:.4f}")
        
        # Save model
        if output_dir:
            joblib_dump(model, output_dir / f"lgbm_fold{fold}.joblib")
    
    mean_smape = np.mean(fold_scores)
    print(f"\nLightGBM CV Complete")
    print(f"   Mean SMAPE: {mean_smape:.4f}")
    print(f"   Std SMAPE:  {np.std(fold_scores):.4f}")
    
    return oof_preds, test_preds.mean(axis=1)


# ============================================================
# XGBOOST MODEL
# ============================================================

def train_xgboost_cv(
    X: np.ndarray,
    y_log: np.ndarray,
    X_test: np.ndarray,
    n_folds: int = 5,
    params: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train XGBoost with K-fold cross-validation.
    
    Returns:
        oof_predictions, test_predictions
    """
    print("\n" + "=" * 60)
    print("XGBOOST TRAINING (K-FOLD CV)")
    print("=" * 60)
    
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'learning_rate': 0.03,
            'max_depth': 7,
            'subsample': 0.7,
            'colsample_bytree': 0.55,
            'lambda': 20.0,
            'alpha': 8.0,
            'tree_method': 'hist',
            'seed': seed
        }
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros((len(X_test), n_folds))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")
        
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_log[train_idx], y_log[val_idx]
        
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=200,
            verbose_eval=250
        )
        
        # Predictions
        oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration+1))
        test_preds[:, fold-1] = model.predict(dtest, iteration_range=(0, model.best_iteration+1))
        
        # Score
        val_smape = smape(from_log(y_val), from_log(oof_preds[val_idx]))
        fold_scores.append(val_smape)
        print(f"Fold {fold} SMAPE: {val_smape:.4f}")
        
        # Save model
        if output_dir:
            model.save_model(str(output_dir / f"xgb_fold{fold}.json"))
    
    mean_smape = np.mean(fold_scores)
    print(f"\nXGBoost CV Complete")
    print(f"   Mean SMAPE: {mean_smape:.4f}")
    print(f"   Std SMAPE:  {np.std(fold_scores):.4f}")
    
    return oof_preds, test_preds.mean(axis=1)


# ============================================================
# CATBOOST MODEL
# ============================================================

def train_catboost_cv(
    X: np.ndarray,
    y_log: np.ndarray,
    X_test: np.ndarray,
    n_folds: int = 5,
    params: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train CatBoost with K-fold cross-validation.
    
    Returns:
        oof_predictions, test_predictions
    """
    print("\n" + "=" * 60)
    print("CATBOOST TRAINING (K-FOLD CV)")
    print("=" * 60)
    
    if params is None:
        params = {
            'loss_function': 'MAE',
            'learning_rate': 0.03,
            'depth': 7,
            'l2_leaf_reg': 30.0,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.7,
            'iterations': 3000,
            'random_seed': seed,
            'verbose': 250
        }
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros((len(X_test), n_folds))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")
        
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_log[train_idx], y_log[val_idx]
        
        train_pool = Pool(X_tr, y_tr)
        val_pool = Pool(X_val, y_val)
        test_pool = Pool(X_test)
        
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        
        # Predictions
        oof_preds[val_idx] = model.predict(val_pool)
        test_preds[:, fold-1] = model.predict(test_pool)
        
        # Score
        val_smape = smape(from_log(y_val), from_log(oof_preds[val_idx]))
        fold_scores.append(val_smape)
        print(f"Fold {fold} SMAPE: {val_smape:.4f}")
        
        # Save model
        if output_dir:
            model.save_model(str(output_dir / f"cat_fold{fold}.cbm"))
    
    mean_smape = np.mean(fold_scores)
    print(f"\nCatBoost CV Complete")
    print(f"   Mean SMAPE: {mean_smape:.4f}")
    print(f"   Std SMAPE:  {np.std(fold_scores):.4f}")
    
    return oof_preds, test_preds.mean(axis=1)


# ============================================================
# ENSEMBLE: OPTIMAL BLENDING
# ============================================================

def optimize_blend_weights(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    weight_range: np.ndarray = np.arange(0.3, 0.51, 0.05)
) -> Dict[str, float]:
    """
    Find optimal ensemble weights via grid search.
    
    Args:
        predictions: dict with model predictions (log-space)
        y_true: true targets (price-space)
        weight_range: range of weights to try
    
    Returns:
        dict with optimal weights
    """
    print("\n" + "=" * 60)
    print("OPTIMIZING ENSEMBLE WEIGHTS")
    print("=" * 60)
    
    model_names = list(predictions.keys())
    assert len(model_names) == 3, "Expected 3 models for blending"
    
    best_smape = float('inf')
    best_weights = None
    
    for w1 in weight_range:
        for w2 in weight_range:
            w3 = 1.0 - w1 - w2
            if w3 < 0.2 or w3 > 0.5:  # reasonable bounds
                continue
            
            # Blend predictions
            blend_log = (
                w1 * predictions[model_names[0]] +
                w2 * predictions[model_names[1]] +
                w3 * predictions[model_names[2]]
            )
            blend = from_log(blend_log)
            
            # Score
            score = smape(y_true, blend)
            
            if score < best_smape:
                best_smape = score
                best_weights = {
                    model_names[0]: w1,
                    model_names[1]: w2,
                    model_names[2]: w3
                }
    
    print(f"\nOptimal weights found:")
    for name, weight in best_weights.items():
        print(f"   {name:12s}: {weight:.2f}")
    print(f"\n   Best SMAPE: {best_smape:.4f}")
    
    return best_weights


def create_ensemble_predictions(
    predictions: Dict[str, np.ndarray],
    weights: Dict[str, float]
) -> np.ndarray:
    """
    Create weighted ensemble predictions.
    
    Args:
        predictions: dict of model predictions (log-space)
        weights: dict of model weights
    
    Returns:
        ensemble predictions (price-space)
    """
    blend_log = sum(
        weights[name] * predictions[name]
        for name in predictions.keys()
    )
    return from_log(blend_log)
