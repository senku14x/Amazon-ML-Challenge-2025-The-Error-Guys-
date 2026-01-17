# -*- coding: utf-8 -*-
"""
Amazon ML Challenge - Product Pricing Pipeline
End-to-end training pipeline for smart product pricing prediction
"""

import re, math, unicodedata, os, gc, json, hashlib
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
BASE_DIR = Path("/content/drive/MyDrive/smart_product_pricing_final")
DATA_DIR = BASE_DIR / "dataset"
SPLIT_DIR = BASE_DIR / "splits"
EMB_DIR = BASE_DIR / "embeddings"
MODEL_DIR = BASE_DIR / "models"
EXP_DIR = BASE_DIR / "experiments"

# Mount Google Drive if in Colab
IN_COLAB = False
try:
    import google.colab
    IN_COLAB = True
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def norm_text(s: Optional[str]) -> str:
    """Normalize text: handle NaN, standardize whitespace"""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()

def norm_unit(u: Optional[str]) -> Optional[str]:
    """Canonicalize measurement units"""
    UNIT_ALIASES = {
        "ml": "ml", "milliliter": "ml", "l": "l", "liter": "l",
        "g": "g", "gram": "g", "kg": "kg", "kilogram": "kg",
        "mg": "mg", "oz": "oz", "ounce": "oz",
        "count": "count", "ct": "count", "pc": "count", "pcs": "count"
    }
    if not u:
        return None
    u = unicodedata.normalize("NFKC", str(u)).lower().strip().replace(".", " ")
    u = re.sub(r"\s+", " ", u)
    return UNIT_ALIASES.get(u, u)

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    out = np.zeros_like(denom)
    out[mask] = np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return out.mean() * 100.0

def robust_read_csv(path: Path, encodings=("utf-8", "latin-1", "iso-8859-1", "cp1252")) -> pd.DataFrame:
    """Try multiple encodings to read CSV"""
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"Could not read {path}")

# ============================================================
# STEP 1: TEXT PARSING AND PACK EXTRACTION
# ============================================================

# Regex patterns
NAME_LINE = re.compile(r"^\s*(?:item\s*name|product\s*name|title)\s*[:\-]\s*(.+?)\s*$", re.I | re.M)
DESC_LINE = re.compile(r"^\s*(?:item\s*description|description|bullet\s*points?)\s*[:\-]\s*(.+?)\s*$", re.I | re.M)
BULLET_LINE = re.compile(r"^(?:[-•*]|\d+[.)])\s*(.+)", re.I | re.M)
COUNT_X_SIZE = re.compile(r"(?P<count>\d{1,3})\s*[xX×*]\s*(?P<size>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z\. ]+)")
SIZE_UNIT = re.compile(r"(?P<size>\d+(?:\.\d+)?)\s*(?P<unit>(?:ml|l|g|kg|mg|oz|count|ct|pc|pcs))", re.I)
COUNT_ONLY = re.compile(r"(?P<count>\d{1,3})\s*(?:count|ct|pcs?|units?)", re.I)

def parse_catalog_cell(text: Optional[str]) -> Dict[str, Optional[str]]:
    """Extract item_name, item_description, and item_pack from catalog text"""
    t = norm_text(text)
    
    # Item name
    m_name = NAME_LINE.search(t)
    item_name = m_name.group(1).strip() if m_name else None
    if not item_name:
        for ln in (ln.strip() for ln in t.split("\n") if ln.strip()):
            if not re.match(r"^(?:item\s*description|description|bullet|ipq|qty)\s*[:\-]", ln, re.I):
                item_name = ln
                break
    
    # Item description
    m_desc = DESC_LINE.search(t)
    if m_desc:
        item_description = m_desc.group(1).strip()
    else:
        bullets = [b.strip() for b in BULLET_LINE.findall(t)]
        item_description = " ".join(dict.fromkeys(bullets)) if bullets else None
    
    if isinstance(item_description, str) and len(item_description) > 1200:
        item_description = item_description[:1200]
    
    # Item pack
    item_pack = None
    for pat in [COUNT_X_SIZE, SIZE_UNIT, COUNT_ONLY]:
        m = pat.search(t)
        if m:
            item_pack = m.group(0).strip()
            break
    
    return {"item_name": item_name, "item_description": item_description, "item_pack": item_pack}

def parse_item_pack(s: Optional[str]) -> Tuple[Optional[float], Optional[str], Optional[int]]:
    """Parse pack string into (value, unit, count)"""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return (None, None, None)
    txt = unicodedata.normalize("NFKC", str(s)).strip().lower()
    
    m = COUNT_X_SIZE.search(txt)
    if m:
        return float(m.group("size")), norm_unit(m.group("unit")), int(m.group("count"))
    
    m = SIZE_UNIT.search(txt)
    if m:
        return float(m.group("size")), norm_unit(m.group("unit")), None
    
    m = COUNT_ONLY.search(txt)
    if m:
        return None, "count", int(m.group("count"))
    
    return (None, None, None)

def parse_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full parsing pipeline: catalog content + pack cleaning"""
    # Parse catalog
    parsed = df["catalog_content"].apply(parse_catalog_cell)
    parsed_df = pd.DataFrame(parsed.tolist(), index=df.index)
    df = pd.concat([df.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)
    
    # Clean pack
    def _parse_pack(cell):
        v, u, c = parse_item_pack(cell)
        clean = None
        if v or c:
            if v and c and u != 'count':
                clean = f"{c}x{v:g} {u}"
            elif v and u:
                clean = f"{v:g} {u}"
            elif c:
                clean = f"{c} count"
        return pd.Series({
            "item_pack_clean": clean,
            "has_pack": int(clean is not None),
            "pack_value": v,
            "pack_unit": u,
            "pack_count": c
        })
    
    pack_parsed = df["item_pack"].apply(_parse_pack)
    return pd.concat([df, pack_parsed], axis=1)

# ============================================================
# STEP 2: TRAIN/VAL SPLIT
# ============================================================

def create_train_val_split(train_df: pd.DataFrame, test_size=0.2):
    """Stratified split by price quantiles"""
    y_price = train_df["price"].astype(float).values
    y_log = np.log1p(y_price)
    bins = pd.qcut(y_log, q=10, labels=False, duplicates='drop')
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    tr_idx, val_idx = next(sss.split(train_df, bins))
    
    # Save indices and targets
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(SPLIT_DIR / "train_idx.npy", tr_idx)
    np.save(SPLIT_DIR / "val_idx.npy", val_idx)
    np.save(SPLIT_DIR / "y_train.npy", y_price[tr_idx])
    np.save(SPLIT_DIR / "y_val.npy", y_price[val_idx])
    np.save(SPLIT_DIR / "y_train_log.npy", y_log[tr_idx])
    np.save(SPLIT_DIR / "y_val_log.npy", y_log[val_idx])
    
    return tr_idx, val_idx

# ============================================================
# STEP 3: MISSING DATA HANDLING
# ============================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with appropriate fill strategies"""
    df = df.copy()
    
    # Text columns
    text_cols = ['catalog_content', 'item_name', 'item_description', 'item_pack_clean']
    for col in text_cols:
        if col in df.columns:
            df[f'{col}_is_missing'] = df[col].isna().astype(int)
            df[col] = df[col].fillna('')
    
    # Numeric columns
    numeric_cols = ['pack_value', 'pack_count']
    for col in numeric_cols:
        if col in df.columns:
            df[f'{col}_is_missing'] = df[col].isna().astype(int)
            fill_value = 0 if 'count' in col else -1
            df[col] = df[col].fillna(fill_value)
    
    # Categorical columns
    categorical_cols = ['pack_unit', 'item_pack']
    for col in categorical_cols:
        if col in df.columns:
            df[f'{col}_is_missing'] = df[col].isna().astype(int)
            df[col] = df[col].fillna('unknown')
    
    # Binary columns
    if 'has_pack' in df.columns:
        df['has_pack'] = df['has_pack'].fillna(0).astype(int)
    
    # URL columns
    if 'image_link' in df.columns:
        df['image_link_is_missing'] = df['image_link'].isna().astype(int)
        df['image_link'] = df['image_link'].fillna('')
    
    return df

# ============================================================
# STEP 4: EMBEDDING GENERATION
# ============================================================

def generate_text_embeddings_e5(texts, model_id="intfloat/e5-large-v2", batch_size=32, max_len=512):
    """Generate E5-large-v2 text embeddings"""
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
    
    model = SentenceTransformer(model_id, device=DEVICE)
    model.max_seq_length = max_len
    model.eval()
    
    texts = ["passage: " + str(t) for t in texts]
    
    @torch.no_grad()
    def encode_batch(batch_texts):
        out = []
        for i in tqdm(range(0, len(batch_texts), batch_size), desc="E5 encoding"):
            batch = batch_texts[i:i+batch_size]
            emb = model.encode(batch, batch_size=len(batch), device=DEVICE,
                             convert_to_numpy=True, normalize_embeddings=True,
                             show_progress_bar=False)
            out.append(emb)
        return np.vstack(out)
    
    return encode_batch(texts)

def generate_image_embeddings_clip(df, cache_dir, model_arch="ViT-L-14", batch_size=192):
    """Generate CLIP image embeddings from cached images"""
    import open_clip
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
    
    model, _, preprocess = open_clip.create_model_and_transforms(model_arch, pretrained="openai")
    model = model.to(DEVICE).eval().half()
    
    class ImgDataset(Dataset):
        def __init__(self, df, cache_dir, preprocess):
            self.paths = [(cache_dir / (str(row["image_link"]) if str(row["image_link"]).endswith(".jpg")
                          else (hashlib.md5(str(row["image_link"]).encode()).hexdigest() + ".jpg")))
                         for _, row in df.iterrows()]
            self.preprocess = preprocess
        
        def __len__(self): return len(self.paths)
        
        def __getitem__(self, i):
            p = self.paths[i]
            if not p.exists():
                img = Image.new("RGB", (224, 224), (255, 255, 255))
            else:
                try:
                    img = Image.open(p).convert("RGB")
                except:
                    img = Image.new("RGB", (224, 224), (255, 255, 255))
            return self.preprocess(img)
    
    dl = DataLoader(ImgDataset(df, cache_dir, preprocess), batch_size=batch_size,
                   num_workers=8, pin_memory=True, shuffle=False)
    
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dl, desc="CLIP encoding"):
            batch = batch.to(DEVICE, dtype=torch.float16)
            z = model.encode_image(batch)
            z = F.normalize(z.float(), dim=-1).cpu().numpy()
            embeddings.append(z)
    
    return np.vstack(embeddings)

# ============================================================
# STEP 5: FEATURE FUSION & TRAINING
# ============================================================

def extract_pack_features(df):
    """Extract pack-related features"""
    pack_cols_num = [c for c in ["pack_value", "pack_count"] if c in df.columns]
    pack_cols_bin = [c for c in ["has_pack"] if c in df.columns]
    
    Xn = df[pack_cols_num].astype("float32") if pack_cols_num else pd.DataFrame(index=df.index)
    Xb = df[pack_cols_bin].astype("float32") if pack_cols_bin else pd.DataFrame(index=df.index)
    
    if "pack_unit" in df.columns:
        vc = df["pack_unit"].fillna("unknown").astype(str)
        oh = pd.get_dummies(vc, prefix="unit", dtype="float32") if vc.nunique() <= 25 else pd.DataFrame(index=df.index)
    else:
        oh = pd.DataFrame(index=df.index)
    
    return pd.concat([Xn, Xb, oh], axis=1)

def train_lightgbm(X_train, y_train_log, X_val, y_val_log):
    """Train LightGBM regressor"""
    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.02,
        "num_leaves": 21,
        "min_data_in_leaf": 600,
        "feature_fraction": 0.4,
        "bagging_fraction": 0.6,
        "bagging_freq": 1,
        "lambda_l1": 25.0,
        "lambda_l2": 50.0,
        "verbosity": -1,
        "seed": SEED
    }
    
    ds_tr = lgb.Dataset(X_train, label=y_train_log)
    ds_val = lgb.Dataset(X_val, label=y_val_log, reference=ds_tr)
    
    gbm = lgb.train(params, ds_tr, valid_sets=[ds_tr, ds_val],
                   valid_names=["train", "val"], num_boost_round=10000,
                   callbacks=[lgb.early_stopping(100), lgb.log_evaluation(250)])
    
    return gbm

def train_xgboost(X_train, y_train_log, X_val, y_val_log):
    """Train XGBoost regressor"""
    params = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "learning_rate": 0.02,
        "max_depth": 5,
        "min_child_weight": 800,
        "subsample": 0.6,
        "colsample_bytree": 0.4,
        "reg_alpha": 25.0,
        "reg_lambda": 50.0,
        "tree_method": "hist",
        "seed": SEED,
        "verbosity": 0
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train_log)
    dval = xgb.DMatrix(X_val, label=y_val_log)
    
    gbm = xgb.train(params, dtrain, num_boost_round=10000,
                   evals=[(dtrain, "train"), (dval, "val")],
                   early_stopping_rounds=100, verbose_eval=250)
    
    return gbm

def train_catboost(X_train, y_train_log, X_val, y_val_log):
    """Train CatBoost regressor"""
    params = {
        "loss_function": "MAE",
        "learning_rate": 0.02,
        "depth": 5,
        "min_data_in_leaf": 800,
        "subsample": 0.6,
        "rsm": 0.4,
        "l2_leaf_reg": 50.0,
        "random_seed": SEED,
        "verbose": 250,
        "early_stopping_rounds": 100,
        "iterations": 10000
    }
    
    train_pool = Pool(X_train, y_train_log)
    val_pool = Pool(X_val, y_val_log)
    
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=250)
    
    return model

# ============================================================
# STEP 6: TRANSFORMER FUSION MODEL
# ============================================================

class CrossModalTransformer(nn.Module):
    def __init__(self, d_in=128, d_model=256, n_heads=4, n_layers=2, ff_dim=512, dropout=0.1):
        super().__init__()
        self.projector = nn.Linear(d_in, d_model)
        self.mod_embed = nn.Embedding(4, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                          dim_feedforward=ff_dim, dropout=dropout,
                                          activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, x):
        B, T, _ = x.shape
        x = self.projector(x)
        mod_ids = torch.arange(0, 4, device=x.device).view(1, 4)
        x = x + self.mod_embed(mod_ids)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)

# ============================================================
# STEP 7: ENSEMBLE & SUBMISSION
# ============================================================

def create_weighted_blend(gbm_pred, transformer_pred, w_gbm=0.5):
    """Create weighted ensemble of GBM and Transformer predictions"""
    w_tr = 1 - w_gbm
    blend = w_gbm * gbm_pred + w_tr * transformer_pred
    return blend

def generate_submission(predictions, sample_ids, output_path):
    """Generate final submission CSV"""
    submission = pd.DataFrame({
        "sample_id": sample_ids,
        "price": predictions
    })
    submission.to_csv(output_path, index=False)
    print(f"Submission saved: {output_path}")
    return submission

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("Starting Amazon ML Challenge Pipeline")
    
    # Step 1: Load and parse data
    print("\n[1/7] Loading and parsing data")
    train_raw = robust_read_csv(DATA_DIR / "train.csv")
    test_raw = robust_read_csv(DATA_DIR / "test.csv")
    
    train_clean = parse_and_clean_data(train_raw)
    test_clean = parse_and_clean_data(test_raw)
    
    # Step 2: Create train/val split
    print("\n[2/7] Creating train/val split")
    tr_idx, val_idx = create_train_val_split(train_clean)
    
    # Step 3: Handle missing data
    print("\n[3/7] Handling missing values")
    df_train = handle_missing_values(train_clean.iloc[tr_idx])
    df_val = handle_missing_values(train_clean.iloc[val_idx])
    df_test = handle_missing_values(test_clean)
    
    # Save processed data
    df_train.to_parquet(SPLIT_DIR / "df_train_processed.parquet")
    df_val.to_parquet(SPLIT_DIR / "df_val_processed.parquet")
    df_test.to_parquet(SPLIT_DIR / "df_test_processed.parquet")
    
    print("Data preprocessing complete")

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
