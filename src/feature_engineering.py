"""
Feature engineering: cross-modal interactions and structured features.

Cross-modal features:
- Cosine similarity between text/image embeddings
- Norm ratios and magnitude differences

Structured features:
- Pack quantity/value interactions
- Text complexity metrics
- Keyword flags
- Numeric extraction
"""

import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import norm


# ============================================================
# CROSS-MODAL INTERACTIONS
# ============================================================

def safe_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity with numerical stability."""
    denom = (norm(A, axis=1) * norm(B, axis=1)) + 1e-8
    return np.sum(A * B, axis=1) / denom


def create_cross_modal_features(
    A_train: np.ndarray,
    A_val: np.ndarray,
    A_test: np.ndarray,
    B_train: np.ndarray,
    B_val: np.ndarray,
    B_test: np.ndarray,
    name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Create interaction features between two modalities.
    
    Features:
        - Cosine similarity
        - Norm of A
        - Norm of B
        - Ratio of norms (A/B)
    
    Returns:
        X_train, X_val, X_test, feature_names
    """
    # Cosine similarity
    cos_train = safe_cosine(A_train, B_train)
    cos_val = safe_cosine(A_val, B_val)
    cos_test = safe_cosine(A_test, B_test)
    
    # Norms
    norm_A_train = norm(A_train, axis=1)
    norm_A_val = norm(A_val, axis=1)
    norm_A_test = norm(A_test, axis=1)
    
    norm_B_train = norm(B_train, axis=1)
    norm_B_val = norm(B_val, axis=1)
    norm_B_test = norm(B_test, axis=1)
    
    # Ratios
    ratio_train = norm_A_train / (norm_B_train + 1e-8)
    ratio_val = norm_A_val / (norm_B_val + 1e-8)
    ratio_test = norm_A_test / (norm_B_test + 1e-8)
    
    # Stack features
    X_train = np.vstack([cos_train, norm_A_train, norm_B_train, ratio_train]).T
    X_val = np.vstack([cos_val, norm_A_val, norm_B_val, ratio_val]).T
    X_test = np.vstack([cos_test, norm_A_test, norm_B_test, ratio_test]).T
    
    feature_names = [
        f"{name}_cos",
        f"{name}_normA",
        f"{name}_normB",
        f"{name}_ratio"
    ]
    
    return X_train, X_val, X_test, feature_names


def build_all_cross_modal_features(
    embeddings: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Build cross-modal features for all modality pairs.
    
    Args:
        embeddings: dict with keys like 'e5_train', 'clip_train', etc.
    
    Returns:
        X_train, X_val, X_test, feature_names
    """
    print("\n" + "=" * 60)
    print("CROSS-MODAL FEATURE ENGINEERING")
    print("=" * 60)
    
    X_train_list, X_val_list, X_test_list, all_names = [], [], [], []
    
    # Define modality pairs
    pairs = []
    if 'e5_train' in embeddings and 'clip_train' in embeddings:
        pairs.append(('e5', 'clip', 'e5_clip'))
    if 'deberta_train' in embeddings and 'clip_train' in embeddings:
        pairs.append(('deberta', 'clip', 'deb_clip'))
    if 'dino_train' in embeddings and 'e5_train' in embeddings:
        pairs.append(('dino', 'e5', 'dino_e5'))
    
    # Generate features for each pair
    for mod_a, mod_b, name in pairs:
        print(f"\nCreating {name} interactions...")
        X_tr, X_val, X_te, names = create_cross_modal_features(
            embeddings[f'{mod_a}_train'],
            embeddings[f'{mod_a}_val'],
            embeddings[f'{mod_a}_test'],
            embeddings[f'{mod_b}_train'],
            embeddings[f'{mod_b}_val'],
            embeddings[f'{mod_b}_test'],
            name
        )
        X_train_list.append(X_tr)
        X_val_list.append(X_val)
        X_test_list.append(X_te)
        all_names.extend(names)
        print(f"   Added {len(names)} features")
    
    # Concatenate all
    if X_train_list:
        X_train = np.concatenate(X_train_list, axis=1)
        X_val = np.concatenate(X_val_list, axis=1)
        X_test = np.concatenate(X_test_list, axis=1)
    else:
        n_train = embeddings[list(embeddings.keys())[0]].shape[0]
        n_val = embeddings[list(embeddings.keys())[1]].shape[0]
        n_test = embeddings[list(embeddings.keys())[2]].shape[0]
        X_train = np.zeros((n_train, 0))
        X_val = np.zeros((n_val, 0))
        X_test = np.zeros((n_test, 0))
    
    print(f"\nCross-modal features created:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape}")
    print(f"   Test:  {X_test.shape}")
    
    return X_train, X_val, X_test, all_names


# ============================================================
# PACK FEATURES
# ============================================================

def extract_pack_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract engineered features from pack quantities."""
    out = pd.DataFrame(index=df.index)
    
    # Clip extreme values
    out['pack_count'] = df['pack_count'].clip(0, 1000)
    out['pack_value'] = df['pack_value'].clip(0, 10000)
    
    # Binary flag
    out['is_single_pack'] = (out['pack_count'] == 1).astype(int)
    
    # Interactions
    out['pack_count_x_value'] = out['pack_count'] * out['pack_value']
    out['log_count_div_value'] = np.log1p(
        out['pack_count'] / (out['pack_value'] + 1e-6)
    )
    
    return out


# ============================================================
# TEXT COMPLEXITY FEATURES
# ============================================================

def compute_text_complexity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute text complexity metrics from item_description."""
    txt = df['item_description'].fillna('')
    
    out = pd.DataFrame(index=df.index)
    out['desc_len'] = txt.str.len()
    out['desc_words'] = txt.str.split().apply(len)
    out['uppercase_ratio'] = txt.apply(
        lambda s: sum(1 for c in s if c.isupper()) / (len(s) + 1e-6)
    )
    out['punct_density'] = txt.apply(
        lambda s: sum(c in ".,;:!?" for c in s) / (len(s) + 1e-6)
    )
    out['digit_ratio'] = txt.apply(
        lambda s: sum(c.isdigit() for c in s) / (len(s) + 1e-6)
    )
    
    return out


# ============================================================
# NUMERIC EXTRACTION
# ============================================================

NUM_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s?(kg|g|l|ml|oz|inch|cm|mm|w|v|mah|gb|tb|l)?",
    re.IGNORECASE
)


def extract_numbers_from_text(text: str) -> dict:
    """Extract all numbers from description text."""
    nums = [float(x[0]) for x in NUM_PATTERN.findall(str(text))]
    
    return {
        'n_numbers': len(nums),
        'num_mean': np.mean(nums) if nums else 0,
        'num_std': np.std(nums) if nums else 0,
        'num_max': np.max(nums) if nums else 0
    }


def compute_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric features from descriptions."""
    feats = df['item_description'].fillna('').apply(extract_numbers_from_text)
    return pd.DataFrame(feats.tolist(), index=df.index)


# ============================================================
# KEYWORD FLAGS
# ============================================================

COMMON_KEYWORDS = [
    'premium', 'organic', 'refill', 'combo', 'mini', 'xl',
    'wireless', 'stainless', 'imported', 'pack of', 'set of'
]


def keyword_flags(text: str) -> list:
    """Check for presence of common keywords."""
    t = str(text).lower()
    return [int(kw in t) for kw in COMMON_KEYWORDS]


def compute_keyword_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary flags for common keywords."""
    flags = np.vstack(df['item_name'].fillna('').apply(keyword_flags))
    return pd.DataFrame(
        flags,
        columns=[f"kw_{kw.replace(' ', '_')}" for kw in COMMON_KEYWORDS],
        index=df.index
    )


# ============================================================
# BRAND FREQUENCY
# ============================================================

def compute_brand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract brand token and compute frequency."""
    out = pd.DataFrame(index=df.index)
    
    # Extract first capitalized word as brand proxy
    brands = df['item_name'].str.extract(r"^([A-Z][A-Za-z0-9&\-]+)")
    out['brand_token'] = brands[0].fillna('UNKNOWN')
    
    # Map to frequency
    brand_counts = out['brand_token'].value_counts()
    out['brand_freq'] = out['brand_token'].map(brand_counts)
    
    # Drop token (keep only frequency)
    out.drop(columns=['brand_token'], inplace=True)
    
    return out


# ============================================================
# TIER-1 FEATURE BUILDER
# ============================================================

def build_tier1_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all Tier-1 structured features.
    
    Returns:
        DataFrame with engineered features
    """
    print("\n" + "=" * 60)
    print("TIER-1 FEATURE ENGINEERING")
    print("=" * 60)
    
    features = []
    
    # Pack features
    print("\nExtracting pack features...")
    pack_feats = extract_pack_features(df)
    features.append(pack_feats)
    print(f"   Added {len(pack_feats.columns)} features")
    
    # Text complexity
    print("Computing text complexity...")
    text_feats = compute_text_complexity(df)
    features.append(text_feats)
    print(f"   Added {len(text_feats.columns)} features")
    
    # Numeric extraction
    print("Extracting numbers...")
    num_feats = compute_numeric_features(df)
    features.append(num_feats)
    print(f"   Added {len(num_feats.columns)} features")
    
    # Keyword flags
    print("Creating keyword flags...")
    kw_feats = compute_keyword_features(df)
    features.append(kw_feats)
    print(f"   Added {len(kw_feats.columns)} features")
    
    # Brand frequency
    print("Computing brand frequency...")
    brand_feats = compute_brand_features(df)
    features.append(brand_feats)
    print(f"   Added {len(brand_feats.columns)} features")
    
    # Concatenate all
    result = pd.concat(features, axis=1)
    
    print(f"\nTier-1 features complete:")
    print(f"   Total features: {len(result.columns)}")
    
    return result


# ============================================================
# FINAL FEATURE FUSION
# ============================================================

def fuse_all_features(
    embeddings_dict: dict,
    cross_modal_features: Tuple[np.ndarray, np.ndarray, np.ndarray],
    tier1_train: pd.DataFrame,
    tier1_val: pd.DataFrame,
    tier1_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Fuse all feature types into final feature matrices.
    
    Args:
        embeddings_dict: dict with PCA-reduced embeddings
        cross_modal_features: (X_train, X_val, X_test) tuple
        tier1_train/val/test: Tier-1 structured features
    
    Returns:
        X_train_final, X_val_final, X_test_final, feature_names
    """
    print("\n" + "=" * 60)
    print("FINAL FEATURE FUSION")
    print("=" * 60)
    
    # Concatenate embeddings
    emb_train_list, emb_val_list, emb_test_list = [], [], []
    emb_names = []
    
    for key in ['e5', 'deberta', 'clip', 'dino']:
        if f'{key}_train' in embeddings_dict:
            emb_train_list.append(embeddings_dict[f'{key}_train'])
            emb_val_list.append(embeddings_dict[f'{key}_val'])
            emb_test_list.append(embeddings_dict[f'{key}_test'])
            n_dims = embeddings_dict[f'{key}_train'].shape[1]
            emb_names.extend([f'{key}_pca{i}' for i in range(n_dims)])
    
    X_emb_train = np.concatenate(emb_train_list, axis=1)
    X_emb_val = np.concatenate(emb_val_list, axis=1)
    X_emb_test = np.concatenate(emb_test_list, axis=1)
    
    # Add cross-modal features
    X_cross_train, X_cross_val, X_cross_test = cross_modal_features
    
    # Add Tier-1 features
    X_tier1_train = tier1_train.values
    X_tier1_val = tier1_val.values
    X_tier1_test = tier1_test.values
    
    # Final concatenation
    X_train_final = np.concatenate([
        X_emb_train,
        X_cross_train,
        X_tier1_train
    ], axis=1)
    
    X_val_final = np.concatenate([
        X_emb_val,
        X_cross_val,
        X_tier1_val
    ], axis=1)
    
    X_test_final = np.concatenate([
        X_emb_test,
        X_cross_test,
        X_tier1_test
    ], axis=1)
    
    # Feature names
    cross_names = [f'cross_{i}' for i in range(X_cross_train.shape[1])]
    all_names = emb_names + cross_names + list(tier1_train.columns)
    
    print(f"\nFeature fusion complete:")
    print(f"   Embeddings:   {X_emb_train.shape[1]} features")
    print(f"   Cross-modal:  {X_cross_train.shape[1]} features")
    print(f"   Tier-1:       {X_tier1_train.shape[1]} features")
    print(f"   Total:        {X_train_final.shape[1]} features")
    print(f"\n   Train shape:  {X_train_final.shape}")
    print(f"   Val shape:    {X_val_final.shape}")
    print(f"   Test shape:   {X_test_final.shape}")
    
    return X_train_final, X_val_final, X_test_final, all_names
