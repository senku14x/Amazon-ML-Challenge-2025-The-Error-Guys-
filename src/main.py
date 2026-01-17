"""
Main pipeline orchestrator for Amazon ML Challenge.

Usage:
    python main.py --config config.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PathConfig, ModelConfig, EmbeddingConfig
from src.utils import log, save_json
from src.preprocessing import preprocess_pipeline
from src.embeddings import (
    generate_e5_embeddings,
    generate_deberta_embeddings,
    generate_clip_embeddings,
    apply_pca_reduction
)
from src.feature_engineering import (
    build_all_cross_modal_features,
    build_tier1_features,
    fuse_all_features
)
from src.modeling import (
    train_lightgbm_cv,
    train_xgboost_cv,
    train_catboost_cv,
    optimize_blend_weights,
    create_ensemble_predictions,
    to_log,
    from_log
)


def main(args):
    """Run full pipeline."""
    
    # ============================================================
    # SETUP
    # ============================================================
    log("=" * 60)
    log("AMAZON ML CHALLENGE - FULL PIPELINE")
    log("=" * 60)
    
    paths = PathConfig.from_base(args.base_dir)
    model_cfg = ModelConfig()
    emb_cfg = EmbeddingConfig()
    
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # STEP 1: PREPROCESSING
    # ============================================================
    if not args.skip_preprocessing:
        log("\nSTEP 1: Preprocessing")
        preprocess_pipeline(
            train_path=paths.data_dir / "train.csv",
            test_path=paths.data_dir / "test.csv",
            output_dir=paths.output_dir / "processed",
            test_size=0.2,
            random_state=model_cfg.seed
        )
    
    # Load processed data
    df_train = pd.read_parquet(paths.output_dir / "processed" / "df_train_processed.parquet")
    df_val = pd.read_parquet(paths.output_dir / "processed" / "df_val_processed.parquet")
    df_test = pd.read_parquet(paths.output_dir / "processed" / "df_test_processed.parquet")
    
    y_train = np.load(paths.output_dir / "processed" / "y_train.npy")
    y_val = np.load(paths.output_dir / "processed" / "y_val.npy")
    y_train_log = to_log(y_train)
    y_val_log = to_log(y_val)
    
    # ============================================================
    # STEP 2: EMBEDDINGS
    # ============================================================
    if not args.skip_embeddings:
        log("\nSTEP 2: Generating Embeddings")
        
        # E5
        X_e5_tr, X_e5_val, X_e5_te = generate_e5_embeddings(
            df_train, df_val, df_test,
            output_dir=paths.embeddings_dir / "e5",
            model_id=emb_cfg.e5_model,
            max_length=emb_cfg.e5_max_length,
            batch_size=emb_cfg.e5_batch_size,
            device=emb_cfg.device
        )
        
        # Apply PCA
        X_e5_tr, X_e5_val, X_e5_te, _ = apply_pca_reduction(
            X_e5_tr, X_e5_val, X_e5_te,
            n_components=emb_cfg.pca_components,
            output_dir=paths.embeddings_dir / "e5",
            name="e5"
        )
        
        # CLIP
        X_clip_tr, X_clip_val, X_clip_te = generate_clip_embeddings(
            df_train, df_val, df_test,
            image_cache_dir=paths.base_dir / "images_cache",
            output_dir=paths.embeddings_dir / "clip",
            model_arch=emb_cfg.clip_model,
            pretrained=emb_cfg.clip_pretrained,
            batch_size=emb_cfg.clip_batch_size,
            device=emb_cfg.device
        )
        
        # Apply PCA
        X_clip_tr, X_clip_val, X_clip_te, _ = apply_pca_reduction(
            X_clip_tr, X_clip_val, X_clip_te,
            n_components=emb_cfg.pca_components,
            output_dir=paths.embeddings_dir / "clip",
            name="clip"
        )
    
    else:
        # Load existing embeddings
        log("\nLoading existing embeddings...")
        X_e5_tr = np.load(paths.embeddings_dir / "e5" / "X_tr_e5_pca128.npy")
        X_e5_val = np.load(paths.embeddings_dir / "e5" / "X_val_e5_pca128.npy")
        X_e5_te = np.load(paths.embeddings_dir / "e5" / "X_te_e5_pca128.npy")
        
        X_clip_tr = np.load(paths.embeddings_dir / "clip" / "X_tr_clip_pca128.npy")
        X_clip_val = np.load(paths.embeddings_dir / "clip" / "X_val_clip_pca128.npy")
        X_clip_te = np.load(paths.embeddings_dir / "clip" / "X_te_clip_pca128.npy")
    
    # ============================================================
    # STEP 3: FEATURE ENGINEERING
    # ============================================================
    log("\nSTEP 3: Feature Engineering")
    
    # Cross-modal features
    embeddings_dict = {
        'e5_train': X_e5_tr, 'e5_val': X_e5_val, 'e5_test': X_e5_te,
        'clip_train': X_clip_tr, 'clip_val': X_clip_val, 'clip_test': X_clip_te
    }
    
    X_cross_tr, X_cross_val, X_cross_te, cross_names = build_all_cross_modal_features(
        embeddings_dict
    )
    
    # Tier-1 structured features
    tier1_train = build_tier1_features(df_train)
    tier1_val = build_tier1_features(df_val)
    tier1_test = build_tier1_features(df_test)
    
    # Final fusion
    X_train, X_val, X_test, feature_names = fuse_all_features(
        {'e5_train': X_e5_tr, 'e5_val': X_e5_val, 'e5_test': X_e5_te,
         'clip_train': X_clip_tr, 'clip_val': X_clip_val, 'clip_test': X_clip_te},
        (X_cross_tr, X_cross_val, X_cross_te),
        tier1_train, tier1_val, tier1_test
    )
    
    # Combine train+val for final training
    X_full = np.vstack([X_train, X_val])
    y_full_log = np.concatenate([y_train_log, y_val_log])
    y_full = np.concatenate([y_train, y_val])
    
    # ============================================================
    # STEP 4: MODEL TRAINING
    # ============================================================
    log("\nSTEP 4: Training Models")
    
    model_output_dir = paths.output_dir / "models"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # LightGBM
    oof_lgb, test_lgb = train_lightgbm_cv(
        X_full, y_full_log, X_test,
        n_folds=model_cfg.n_folds,
        params=model_cfg.lgb_params,
        output_dir=model_output_dir,
        seed=model_cfg.seed
    )
    
    # XGBoost
    oof_xgb, test_xgb = train_xgboost_cv(
        X_full, y_full_log, X_test,
        n_folds=model_cfg.n_folds,
        params=model_cfg.xgb_params,
        output_dir=model_output_dir,
        seed=model_cfg.seed
    )
    
    # CatBoost
    oof_cat, test_cat = train_catboost_cv(
        X_full, y_full_log, X_test,
        n_folds=model_cfg.n_folds,
        params=model_cfg.cat_params,
        output_dir=model_output_dir,
        seed=model_cfg.seed
    )
    
    # ============================================================
    # STEP 5: ENSEMBLE OPTIMIZATION
    # ============================================================
    log("\nSTEP 5: Ensemble Optimization")
    
    predictions_log = {
        'lgb': oof_lgb,
        'xgb': oof_xgb,
        'cat': oof_cat
    }
    
    optimal_weights = optimize_blend_weights(predictions_log, y_full)
    
    # Create final ensemble predictions
    test_predictions_log = {
        'lgb': test_lgb,
        'xgb': test_xgb,
        'cat': test_cat
    }
    
    test_ensemble = create_ensemble_predictions(
        test_predictions_log,
        optimal_weights
    )
    
    # ============================================================
    # STEP 6: SAVE SUBMISSION
    # ============================================================
    log("\nSTEP 6: Creating Submission")
    
    submission = pd.DataFrame({
        'sample_id': df_test['sample_id'],
        'price': test_ensemble
    })
    
    submission_path = paths.output_dir / "submission_final.csv"
    submission.to_csv(submission_path, index=False)
    
    log(f"\nPIPELINE COMPLETE!")
    log(f"   Submission saved: {submission_path}")
    log(f"   Mean predicted price: ${test_ensemble.mean():.2f}")
    log(f"   Price range: ${test_ensemble.min():.2f} - ${test_ensemble.max():.2f}")
    
    # Save metadata
    save_json({
        'optimal_weights': optimal_weights,
        'feature_count': len(feature_names),
        'n_folds': model_cfg.n_folds,
        'seed': model_cfg.seed
    }, paths.output_dir / "run_metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_dir',
        type=str,
        default=".",
        help="Base directory for data and outputs"
    )
    parser.add_argument(
        '--skip_preprocessing',
        action='store_true',
        help="Skip preprocessing step"
    )
    parser.add_argument(
        '--skip_embeddings',
        action='store_true',
        help="Skip embedding generation (load existing)"
    )
    
    args = parser.parse_args()
    main(args)
