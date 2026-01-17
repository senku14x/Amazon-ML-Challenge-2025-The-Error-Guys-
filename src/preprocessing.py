"""
Data preprocessing and catalog parsing for Amazon ML Challenge.

Handles:
- Raw catalog text parsing (item name, description, pack info)
- Unit normalization and quantity extraction
- Train/val splitting with stratification
"""

import re
import math
import unicodedata
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def norm_text(s: Optional[str]) -> str:
    """Normalize text: Unicode, whitespace, newlines."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


# ============================================================
# REGEX PATTERNS
# ============================================================

NAME_LINE = re.compile(
    r"^\s*(?:item\s*name|product\s*name|item\s*title|title|name)\s*[:\-]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

DESC_LINE = re.compile(
    r"^\s*(?:item\s*description|description|bullet\s*points?)\s*[:\-]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

BULLET_LINE = re.compile(r"^(?:[-•*]|\d+[.)])\s*(.+)", re.IGNORECASE | re.MULTILINE)

ANY_LABEL = re.compile(
    r"(?:^|\n)\s*(?:item\s*name|product\s*name|item\s*title|title|name|"
    r"item\s*description|description|bullet\s*points?|"
    r"item\s*pack\s*quantity|ipq|quantity|qty|value|size|unit|uom)\s*[:\-]",
    re.IGNORECASE | re.MULTILINE,
)

COUNT_X_SIZE = re.compile(
    r"(?P<count>\d{1,3})\s*[xX×*]\s*(?P<size>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z\. ]+)"
)
SIZE_UNIT = re.compile(
    r"(?P<size>\d+(?:\.\d+)?)\s*(?P<unit>(?:ml|l|g|kg|mg|oz|fl\.?\s*oz|count|ct|pc|pcs))",
    re.IGNORECASE
)
COUNT_ONLY = re.compile(r"(?P<count>\d{1,3})\s*(?:count|ct|pcs?|units?)", re.IGNORECASE)


# ============================================================
# UNIT CANONICALIZATION
# ============================================================

UNIT_ALIASES = {
    "ml": "ml", "milliliter": "ml", "millilitre": "ml", "milliliters": "ml", "millilitres": "ml",
    "l": "l", "liter": "l", "litre": "l", "liters": "l", "litres": "l",
    "cl": "cl", "fl oz": "fl oz", "fluid ounce": "fl oz", "fluid ounces": "fl oz",
    "g": "g", "gram": "g", "grams": "g", "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "mg": "mg", "oz": "oz", "ounce": "oz", "ounces": "oz",
    "count": "count", "ct": "count", "pc": "count", "pcs": "count", 
    "piece": "count", "pieces": "count", "unit": "count", "units": "count",
}


def norm_unit(u: Optional[str]) -> Optional[str]:
    """Canonicalize unit strings."""
    if not u:
        return None
    u = unicodedata.normalize("NFKC", str(u)).lower().strip().replace(".", " ")
    u = re.sub(r"\s+", " ", u)
    return UNIT_ALIASES.get(u, u)


# ============================================================
# CATALOG PARSING
# ============================================================

def parse_catalog_cell(text: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Extract structured fields from semi-structured catalog text.
    
    Returns:
        dict with keys: item_name, item_description, item_pack
    """
    t = norm_text(text)

    # --- Item name ---
    m_name = NAME_LINE.search(t)
    if m_name:
        item_name = m_name.group(1).strip()
    else:
        # Fallback: first non-label line
        item_name = None
        for ln in (ln.strip() for ln in t.split("\n") if ln.strip()):
            if not re.match(
                r"^(?:item\s*description|description|bullet\s*points?|"
                r"item\s*pack\s*quantity|ipq|quantity|qty|value|size|unit|uom)\s*[:\-]",
                ln, re.IGNORECASE,
            ):
                item_name = ln
                break

    # --- Item description ---
    m_desc = DESC_LINE.search(t)
    if m_desc:
        item_description = m_desc.group(1).strip()
    else:
        bullets = [b.strip() for b in BULLET_LINE.findall(t)]
        if bullets:
            item_description = " ".join(dict.fromkeys(bullets))
        else:
            item_description = None
            if item_name:
                try:
                    start = t.lower().find(item_name.lower()) + len(item_name)
                    nxt = ANY_LABEL.search(t[start:])
                    mid = t[start:(start + nxt.start() if nxt else len(t))]
                    cand = [s.strip() for s in re.split(r"(?<=[.?!])\s+|\n", mid) if s.strip()]
                    item_description = " ".join(cand[:8]) if cand else None
                except Exception:
                    item_description = None
    
    if isinstance(item_description, str) and len(item_description) > 1200:
        item_description = item_description[:1200]

    # --- Item pack ---
    item_pack = None
    for pat in [COUNT_X_SIZE, SIZE_UNIT, COUNT_ONLY]:
        m = pat.search(t)
        if m:
            item_pack = m.group(0).strip()
            break

    return {
        "item_name": item_name, 
        "item_description": item_description, 
        "item_pack": item_pack
    }


def parse_dataframe(df: pd.DataFrame, catalog_col: str = "catalog_content") -> pd.DataFrame:
    """Apply catalog parsing to entire DataFrame."""
    parsed = df[catalog_col].apply(parse_catalog_cell)
    parsed_df = pd.DataFrame(parsed.tolist(), index=df.index)
    return pd.concat([df.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)


# ============================================================
# PACK QUANTITY PARSING
# ============================================================

def parse_item_pack(s: Optional[str]) -> Tuple[Optional[float], Optional[str], Optional[int]]:
    """
    Extract (pack_value, pack_unit, pack_count) from pack string.
    
    Examples:
        "6x250ml" → (250.0, "ml", 6)
        "500g" → (500.0, "g", None)
        "12 count" → (None, "count", 12)
    """
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return (None, None, None)
    
    txt = unicodedata.normalize("NFKC", str(s)).strip().lower()
    txt = re.sub(r"[–—]", "-", txt)
    txt = re.sub(r"\s+", " ", txt)

    # Try: count x size
    m = COUNT_X_SIZE.search(txt)
    if m:
        return float(m.group("size")), norm_unit(m.group("unit")), int(m.group("count"))
    
    # Try: size + unit
    m = SIZE_UNIT.search(txt)
    if m:
        return float(m.group("size")), norm_unit(m.group("unit")), None
    
    # Try: count only
    m = COUNT_ONLY.search(txt)
    if m:
        return None, "count", int(m.group("count"))
    
    return (None, None, None)


def clean_pack_column(df: pd.DataFrame) -> pd.DataFrame:
    """Parse item_pack into numeric fields: pack_value, pack_unit, pack_count."""
    def _parse(cell):
        v, u, c = parse_item_pack(cell)
        
        # Reconstruct clean string
        if v or c:
            if v and c and u != 'count':
                clean = f"{c}x{v:g} {u}"
            elif v and u:
                clean = f"{v:g} {u}"
            elif c:
                clean = f"{c} count"
            else:
                clean = None
        else:
            clean = None
        
        return pd.Series({
            "item_pack_clean": clean,
            "has_pack": int(clean is not None),
            "pack_value": v,
            "pack_unit": u,
            "pack_count": c
        })

    parsed = df["item_pack"].apply(_parse)
    return pd.concat([df, parsed], axis=1)


# ============================================================
# MISSING DATA HANDLING
# ============================================================

def handle_missing_values(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    """
    Impute missing values with appropriate defaults and create missing indicators.
    
    Strategy:
        - Text columns → empty string + _is_missing flag
        - Numeric columns → -1 (or 0 for counts) + _is_missing flag
        - Categorical → 'unknown' + _is_missing flag
    """
    df = df.copy()

    # Text columns
    text_columns = ['catalog_content', 'item_name', 'item_description', 'item_pack_clean']
    for col in text_columns:
        if col in df.columns:
            df[f'{col}_is_missing'] = df[col].isna().astype(int)
            df[col] = df[col].fillna('')

    # Numeric columns
    numeric_columns = ['pack_value', 'pack_count']
    for col in numeric_columns:
        if col in df.columns:
            df[f'{col}_is_missing'] = df[col].isna().astype(int)
            fill_value = 0 if 'count' in col else -1
            df[col] = df[col].fillna(fill_value)

    # Categorical columns
    categorical_columns = ['pack_unit', 'item_pack']
    for col in categorical_columns:
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
# TRAIN/VAL SPLIT
# ============================================================

def create_stratified_split(
    df: pd.DataFrame,
    target_col: str = 'price',
    test_size: float = 0.2,
    n_bins: int = 10,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create stratified train/val split based on target quantiles.
    
    Returns:
        train_idx, val_idx (numpy arrays)
    """
    y = df[target_col].astype(float).values
    y_log = np.log1p(y)
    
    # Bin by quantiles for stratification
    bins = pd.qcut(y_log, q=n_bins, labels=False, duplicates='drop')
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(sss.split(df, bins))
    
    return train_idx, val_idx


# ============================================================
# PIPELINE RUNNER
# ============================================================

def preprocess_pipeline(
    train_path: Path,
    test_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Path]:
    """
    Run full preprocessing pipeline.
    
    Returns:
        dict of output file paths
    """
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Load raw data
    print(f"\nLoading: {train_path.name}")
    train_raw = pd.read_csv(train_path)
    print(f"Loading: {test_path.name}")
    test_raw = pd.read_csv(test_path)
    
    # Parse catalog fields
    print("\nParsing catalog text...")
    train_parsed = parse_dataframe(train_raw)
    test_parsed = parse_dataframe(test_raw)
    
    # Clean pack quantities
    print("Extracting pack quantities...")
    train_clean = clean_pack_column(train_parsed)
    test_clean = clean_pack_column(test_parsed)
    
    # Create train/val split
    print("\nCreating stratified split...")
    train_idx, val_idx = create_stratified_split(
        train_clean, 
        target_col='price',
        test_size=test_size,
        random_state=random_state
    )
    
    df_train = train_clean.iloc[train_idx].reset_index(drop=True)
    df_val = train_clean.iloc[val_idx].reset_index(drop=True)
    
    # Handle missing values
    print("\nHandling missing values...")
    df_train = handle_missing_values(df_train, is_test=False)
    df_val = handle_missing_values(df_val, is_test=False)
    df_test = handle_missing_values(test_clean, is_test=True)
    
    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    paths['train'] = output_dir / 'df_train_processed.parquet'
    paths['val'] = output_dir / 'df_val_processed.parquet'
    paths['test'] = output_dir / 'df_test_processed.parquet'
    paths['train_idx'] = output_dir / 'train_idx.npy'
    paths['val_idx'] = output_dir / 'val_idx.npy'
    
    df_train.to_parquet(paths['train'], index=False)
    df_val.to_parquet(paths['val'], index=False)
    df_test.to_parquet(paths['test'], index=False)
    np.save(paths['train_idx'], train_idx)
    np.save(paths['val_idx'], val_idx)
    
    # Save targets
    y_train = df_train['price'].values
    y_val = df_val['price'].values
    np.save(output_dir / 'y_train.npy', y_train)
    np.save(output_dir / 'y_val.npy', y_val)
    np.save(output_dir / 'y_train_log.npy', np.log1p(y_train))
    np.save(output_dir / 'y_val_log.npy', np.log1p(y_val))
    
    print(f"\nPreprocessing complete!")
    print(f"   Train: {len(df_train):,} samples")
    print(f"   Val:   {len(df_val):,} samples")
    print(f"   Test:  {len(df_test):,} samples")
    print(f"   Outputs saved to: {output_dir}")
    
    return paths
