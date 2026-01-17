# Technical Methodology - Amazon ML Challenge 2025

*Detailed documentation of the multimodal price prediction system*

---

## Table of Contents
1. [Data Processing Pipeline](#1-data-processing-pipeline)
2. [Embedding Generation](#2-embedding-generation)
3. [Feature Engineering](#3-feature-engineering)
4. [Modeling Strategy](#4-modeling-strategy)
5. [Ensemble & Optimization](#5-ensemble--optimization)
6. [Validation Framework](#6-validation-framework)

---

## 1. Data Processing Pipeline

### 1.1 Raw Data Challenges

**Input Format:**
```
catalog_content: Semi-structured text with fields like:
  - Item Name: Samsung Galaxy S23 Ultra
  - Description: Premium flagship smartphone...
  - Pack Quantity: 1 unit

image_link: URL to product image
```

**Key Issues:**
- 15-20% missing descriptions
- Inconsistent field formatting (multi-language, varying structures)
- Pack quantities in natural language ("3x500ml", "pack of 12")

### 1.2 Text Parsing Strategy

**Regex-Based Field Extraction:**
```python
NAME_PATTERN = r"^\s*(?:item\s*name|product\s*name|item\s*title|title|name)\s*[:\-]\s*(.+?)\s*$"
DESC_PATTERN = r"^\s*(?:item\s*description|description|bullet\s*points?)\s*[:\-]\s*(.+?)\s*$"
PACK_PATTERN = r"(?P<count>\d{1,3})\s*[xX×*]\s*(?P<size>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z\. ]+)"
```

**Normalization Steps:**
1. Unicode NFKC normalization (`unicodedata.normalize("NFKC", text)`)
2. Whitespace canonicalization  
3. Unit standardization (ml → milliliters, g → grams)
4. Missing value imputation (empty strings for text, -1 for numeric)

**Pack Quantity Extraction:**
- Pattern matching for "NxM unit" (e.g., "6x250ml")
- Fallback extraction: size-only ("500g") or count-only ("pack of 3")
- Structured output: `(pack_value, pack_unit, pack_count)`

**Parse Function:**
```python
def parse_catalog_cell(text):
    # Extract item_name (first non-label line or explicit "Item Name:" field)
    # Extract item_description (bullet points or "Description:" field)
    # Extract item_pack (first match of COUNT_X_SIZE, SIZE_UNIT, or COUNT_ONLY)
    return {"item_name": ..., "item_description": ..., "item_pack": ...}
```

### 1.3 Train/Validation Split

**Stratified by Price Quantiles:**
- 80/20 split using `StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)`
- Binned log-price into 10 quantiles to ensure balanced distribution
- Critical for stable validation metrics given right-skewed price distribution

**Split Preservation:**
- Saved as `train_idx.npy` and `val_idx.npy` for reproducibility
- Targets saved in both raw and log-transformed forms

---

## 2. Embedding Generation

### 2.1 Text Embeddings

#### **E5-Large-v2** (Retrieval-Optimized)
- **Model:** `intfloat/e5-large-v2`
- **Architecture:** 1024-dim sentence embeddings
- **Purpose:** Capture semantic similarity across product categories
- **Input Format:** `"passage: {item_name}. {item_description}"`
- **Normalization:** L2-normalized via `normalize_embeddings=True`
- **PCA Reduction:** 1024 → 128 dims (via `sklearn.decomposition.PCA`)
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-large-v2", device=DEVICE)
model.max_seq_length = 512
embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
```

**Processing Details:**
- Batch size: 32 (optimized for GPU memory)
- Max sequence length: 512 tokens
- Combined `item_name + ". " + item_description` for rich context

#### **DeBERTa-v3-base** (Task-Specific Fine-Tuning)
- **Model:** `microsoft/deberta-v3-base`
- **Fine-tuning Task:** Direct log-price regression on training set
- **Training Config:**
  - 6 epochs, AdamW optimizer (lr=2e-5)
  - Early stopping (patience=2)
  - Batch size: 8 per device
  - Max sequence length: 256 tokens
- **Embedding Extraction:** Mean-pooled hidden states from `.deberta` module (not regression head)
```python
# After fine-tuning, extract embeddings:
hidden_states = model.deberta(**tokens)[0]  # [batch, seq_len, 768]
embeddings = mean_pool(hidden_states, attention_mask)
embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalize
```

**Why Two Text Models?**
- E5: General semantic understanding (category, attributes)
- DeBERTa: Price-specific patterns learned through supervised fine-tuning

### 2.2 Image Embeddings

#### **OpenCLIP ViT-L/14** (Vision-Language Alignment)
- **Model:** `ViT-L-14` with `openai` pretrained weights
- **Architecture:** 768-dim image features from Vision Transformer Large
- **Batch Size:** 192 (A100 GPU, FP16 precision)
- **PCA Reduction:** 768 → 128 dims
```python
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
model = model.to(DEVICE).eval().half()
image_features = model.encode_image(images)  # [batch, 768]
```

**Optimization:**
- FP16 (half precision) for 2x speed + memory savings
- `torch.backends.cudnn.benchmark = True` for kernel auto-tuning
- Prefetch factor: 2 with 8 workers for data loading

#### **DINOv2-base** (Self-Supervised Vision)
- **Model:** `facebook/dinov2-base`
- **Architecture:** Mean-pooled patch tokens → 768-dim
- **Purpose:** Complementary visual features (packaging, aesthetics)
- **PCA Reduction:** 768 → 128 dims
```python
from transformers import AutoModel, AutoImageProcessor

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")
outputs = model(pixel_values=pixel_values)
embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling
```

**Image Handling:**
- Missing images → White placeholder (224×224 RGB)
- All images pre-downloaded to local cache (avoided repeated downloads)
- Batch processing with `DataLoader` for efficiency

### 2.3 PCA Dimensionality Reduction

**Applied to all embedding modalities:**
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=128, svd_solver="auto", random_state=42)
X_reduced = pca.fit_transform(X_full)  # fit on train only
```

**Rationale:**
- Reduces correlation between features
- Speeds up GBM training
- Typically preserves 90-95% of variance

**PCA Models Saved:**
- `pca_e5.joblib`, `pca_deb.joblib`, `pca_clip.joblib`, `pca_dino.joblib`
- Applied consistently to train/val/test splits

---

## 3. Feature Engineering

### 3.1 Cross-Modal Interaction Features

**Motivation:** Text-image alignment signals product quality/authenticity

**Generated Features (per modality pair):**
```python
def safe_cosine(A, B):
    """Cosine similarity with numerical stability"""
    denom = (norm(A, axis=1) * norm(B, axis=1)) + 1e-8
    return np.sum(A * B, axis=1) / denom

def make_interactions(A_tr, A_val, A_te, B_tr, B_val, B_te, name):
    cos_sim = safe_cosine(A, B)
    norm_A = np.linalg.norm(A, axis=1)
    norm_B = np.linalg.norm(B, axis=1)
    ratio = norm_A / (norm_B + 1e-8)
    
    return [cos_sim, norm_A, norm_B, ratio]  # 4 features per pair
```

**Modality Pairs (12 total features):**
- E5 ↔ CLIP (text-image semantic alignment) → 4 features
- DeBERTa ↔ CLIP (price-predictive text ↔ visual packaging) → 4 features
- DINOv2 ↔ E5 (aesthetic features ↔ product description) → 4 features

**Additional Interaction:**
```python
# Pack value × first cosine similarity
pack_cross = pack_value * cos_sim[0]  # captures bulk pricing patterns
```

### 3.2 Structured Features (Tier-1 Engineering)

**Pack-Related Features:**
```python
def extract_pack_features(df):
    pack_count = df["pack_count"].clip(0, 1000)
    pack_value = df["pack_value"].clip(0, 10000)
    
    features = {
        "is_single_pack": (pack_count == 1).astype(int),
        "pack_count_x_value": pack_count * pack_value,
        "log_count_div_value": np.log1p(pack_count / (pack_value + 1e-6))
    }
    return features
```

**Text Complexity Metrics:**
```python
def text_complexity(df):
    txt = df["item_description"].fillna("")
    
    return {
        "desc_len": txt.str.len(),
        "desc_words": txt.str.split().apply(len),
        "uppercase_ratio": txt.apply(lambda s: sum(c.isupper() for c in s) / (len(s) + 1e-6)),
        "punct_density": txt.apply(lambda s: sum(c in ".,;:!?" for c in s) / (len(s) + 1e-6)),
        "digit_ratio": txt.apply(lambda s: sum(c.isdigit() for c in s) / (len(s) + 1e-6))
    }
```

**Keyword Flags:**
```python
common_keywords = ["premium", "organic", "refill", "combo", "mini", "xl", 
                   "wireless", "stainless", "imported", "pack of", "set of"]

def keyword_flags(text):
    t = str(text).lower()
    return [int(kw in t) for kw in common_keywords]
```

**Numeric Extraction from Descriptions:**
```python
num_pattern = re.compile(r"(\d+(?:\.\d+)?)\s?(kg|g|l|ml|oz|inch|cm|mm|w|v|mah|gb|tb)?", re.I)

def harvest_numbers(text):
    nums = [float(x[0]) for x in num_pattern.findall(str(text))]
    return {
        "n_numbers": len(nums),
        "num_mean": np.mean(nums) if nums else 0,
        "num_std": np.std(nums) if nums else 0,
        "num_max": np.max(nums) if nums else 0
    }
```

**Brand Frequency:**
```python
# Extract first capitalized word as brand proxy
brands = df["item_name"].str.extract(r"^([A-Z][A-Za-z0-9&\-]+)")
brand_counts = brands.value_counts()
df["brand_freq"] = brands.map(brand_counts)
```

### 3.3 Feature Dimensionality

**Before Pruning:**
- PCA embeddings: 128×4 = 512 dims (E5, DeBERTa, CLIP, DINOv2)
- Cross-modal interactions: 12 dims (3 pairs × 4 features)
- Pack features (scaled): 2-3 dims
- Tier-1 engineered: ~25 dims
- **Total: ~550 features**

**After SHAP Pruning:**
```python
# Keep features contributing to top 80% cumulative gain
imp_df["cum_gain"] = imp_df["gain"].cumsum() / imp_df["gain"].sum()
selected = imp_df[imp_df["cum_gain"] <= 0.80]["feature"].tolist()
```
- **Final: ~190 features** (65% reduction)
- Minimal performance loss (~0.3% SMAPE)

---

## 4. Modeling Strategy

### 4.1 Target Transformation

**Log-Space Regression:**
```python
y_log = np.log1p(y)  # log(1 + price) to handle zeros
```

**Rationale:**
- Price distribution heavily right-skewed
- Log transform → approximate normality
- MAE in log-space correlates with SMAPE

**Inverse Transform:**
```python
y_pred = np.expm1(y_log_pred)  # exp(y) - 1
```

### 4.2 Base Model Configurations

#### **LightGBM** (Primary Model)
```python
lgb_params = {
    "objective": "regression_l1",  # MAE loss
    "metric": "mae",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 400,
    "feature_fraction": 0.55,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l1": 8.0,
    "lambda_l2": 20.0,
    "max_depth": -1,
    "verbosity": -1,
    "seed": 42
}
```

**Training:**
- 5-fold cross-validation
- Early stopping: 200 rounds
- Validation monitoring: train + val MAE

#### **XGBoost** (Diversity Model)
```python
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "tree_method": "gpu_hist",
    "learning_rate": 0.03,
    "max_depth": 7,
    "subsample": 0.7,
    "colsample_bytree": 0.55,
    "lambda": 20.0,  # L2
    "alpha": 8.0,    # L1
    "random_state": 42
}
```

#### **CatBoost** (Regularization Focus)
```python
cat_params = {
    "loss_function": "MAE",
    "learning_rate": 0.03,
    "depth": 7,
    "l2_leaf_reg": 30.0,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.7,
    "iterations": 3000,
    "task_type": "GPU",
    "random_seed": 42,
    "verbose": 500
}
```

**Why Three Models?**
- Different tree-building algorithms → uncorrelated errors
- Cross-model correlation ~0.994-0.999 (stable but not redundant)
- Ensemble reduces variance

### 4.3 Pseudo-Labeling Enhancement

**Data Source:**
- Used predictions from previous experiment: `pred_test_lgbmeta.npy`
- Based on stacked ensemble from earlier iteration

**Selection Strategy:**
```python
# High confidence = low prediction variance (single model, so use threshold)
CONFIDENCE_THRESHOLD = 0.05  # in log-space
pseudo_df = test_predictions[predictions_std < CONFIDENCE_THRESHOLD]
```

**Weighted Retraining:**
```python
X_full = np.vstack([X_labeled, X_pseudo])
y_full = np.hstack([y_labeled, y_pseudo])
weights = np.hstack([
    np.ones(len(y_labeled)),      # 1.0 weight for labeled
    np.full(len(y_pseudo), 0.5)   # 0.5 weight for pseudo
])

lgb.Dataset(X_full, label=y_full, weight=weights)
```

**Impact:**
- Added ~75k pseudo-labeled samples
- Conservative 0.5× weight prevents label noise propagation
- Observed ~2% SMAPE improvement on validation

---

## 5. Ensemble & Optimization

### 5.1 Optimal Blending (Grid Search)

**Search Space:**
```python
best_smape = float('inf')
best_weights = None

for w_lgb in np.arange(0.3, 0.51, 0.05):
    for w_xgb in np.arange(0.2, 0.41, 0.05):
        w_cat = 1 - w_lgb - w_xgb
        if w_cat < 0:
            continue
        
        blend = w_lgb * pred_lgb + w_xgb * pred_xgb + w_cat * pred_cat
        score = smape(y_val, blend)
        
        if score < best_smape:
            best_smape = score
            best_weights = (w_lgb, w_xgb, w_cat)
```

**Best Weights (Validation-Optimized):**
- LightGBM: 0.40
- XGBoost: 0.30
- CatBoost: 0.30

**Alternative Tested:**
- Simple average (0.33, 0.33, 0.34): Slightly worse (~0.2% SMAPE increase)
- Meta-learner (ElasticNet): Overfitted on validation

### 5.2 SMAPE Calculation

**Implementation:**
```python
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    out = np.zeros_like(denom)
    out[mask] = np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return out.mean() * 100.0
```

**Formula:** `100 × mean(|pred - true| / ((|pred| + |true|) / 2))`

---

## 6. Validation Framework

### 6.1 Validation Strategy

**K-Fold Cross-Validation:**
- 5 folds, stratified by log-price quantiles
- Each model trained on 4 folds, validated on 1
- Out-of-fold predictions aggregated for final validation score

**Holdout Validation:**
- 20% stratified holdout (fixed `val_idx.npy`)
- Used for final ensemble weight optimization
- Never touched during model development

### 6.2 Validation Checks

**1. Out-of-Fold Consistency**
```python
# OOF predictions from CV
oof_blend = 0.4 * oof_lgb + 0.3 * oof_xgb + 0.3 * oof_cat
oof_mae = mean_absolute_error(y_true, oof_blend)
```

**2. Pseudo-Label Alignment**
```python
# Check agreement with pseudo-labels
pred_test_log = np.log1p(blend_pred)
diff = np.abs(pred_test_log[confident_mask] - y_pseudo_log[confident_mask])
print(f"Mean diff: {diff.mean():.4f} log-points")
```

**3. Distribution Sanity**
```python
# Visual check: train vs test distributions
plt.hist(y_train, bins=50, alpha=0.5, label="Train")
plt.hist(pred_test, bins=50, alpha=0.5, label="Test")

# Numerical check
assert np.abs(y_train.mean() - pred_test.mean()) < 5  # within $5
assert pred_test.min() >= 0  # no negative prices
```

**4. Cross-Model Stability**
```python
# High correlation → stable predictions
corr_lgb_xgb = np.corrcoef(pred_lgb, pred_xgb)[0, 1]
assert corr_lgb_xgb > 0.99, "Models diverging significantly"
```

---

## 7. Computational Resources

**Hardware:**
- Google Colab Pro (A100 40GB GPU)
- 83GB system RAM
- High-RAM runtime

**Training Time (Approximate):**
- Text embedding (E5 + DeBERTa fine-tuning): ~3 hours
- Image embedding (CLIP + DINOv2): ~2 hours
- Feature engineering: ~30 minutes
- GBM training (3 models, 5-fold CV each): ~3 hours
- **Total pipeline: ~8-9 hours**

**Storage:**
- Raw CSV data: 2.1 GB
- Image cache: 15 GB (pre-downloaded)
- Embeddings (NPY files): 4.5 GB
- Model checkpoints (.joblib/.cbm/.json): ~1 GB

---

## 8. Reproducibility

**Environment:**
```
python==3.10
torch==2.1.0+cu118
transformers==4.44.0
sentence-transformers==2.2.2
open-clip-torch==2.23.0
lightgbm==4.1.0
xgboost==2.0.3
catboost==1.2.2
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
```

**Fixed Seeds:**
```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

**Saved Artifacts:**
- `train_idx.npy`, `val_idx.npy` (split indices)
- `y_train_log.npy`, `y_val_log.npy` (targets)
- All PCA models (`pca_*.joblib`)
- All trained GBM models (`lgbm_fold*.joblib`, etc.)

---

## 9. Key Learnings

### What Worked

1. **Heavy Regularization**
   - L1=8-25, L2=20-50 in GBMs
   - Prevented overfitting on 75k samples
   - More effective than increasing data via augmentation

2. **Cross-Modal Features**
   - Text-image cosine similarity captured listing quality
   - Norm ratios detected misaligned or low-effort listings
   - +15% improvement over embeddings alone

3. **Conservative Pseudo-Labeling**
   - Tight confidence threshold (σ < 0.05)
   - 0.5× weight for pseudo-samples
   - Added data without introducing noise

4. **PCA Dimensionality Reduction**
   - 768 → 128 dims per modality
   - Decorrelated features, faster training
   - Minimal information loss (90-95% variance retained)

### What Didn't Work

1. **Deep Neural Network Fusion**
   - Transformer-based cross-modal fusion overfitted
   - Higher variance than simple linear blending
   - Required much longer training time

2. **External Price Data**
   - Scraped competitor prices had distribution mismatch
   - Introduced out-of-distribution bias
   - Simple internal features more reliable

3. **Complex Feature Interactions**
   - Polynomial features (degree=2) added noise
   - Price regime clustering (k-means on embeddings) didn't help
   - Simpler features + strong regularization > feature complexity

4. **Stacking Meta-Learner**
   - ElasticNet/Ridge on model predictions overfitted validation set
   - Grid search blending worked better (simpler, more stable)

---

## 10. Future Work

**Potential Improvements:**

1. **Larger Foundation Models**
   - E5-mistral-7b (4096-dim)
   - CLIP ViT-G/14 (1024-dim)
   - Estimated +1-2% SMAPE gain

2. **Iterative Pseudo-Labeling**
   - Re-train with newly pseudo-labeled samples
   - Curriculum learning (easy → hard samples)
   - Requires careful confidence calibration

3. **Attention-Based Modality Fusion**
   - Learnable weights per modality per sample
   - Current approach: Fixed PCA reduction
   - Could better handle domain-specific patterns

4. **Category-Specific Models**
   - Separate models for electronics, groceries, apparel
   - Specialized features per category
   - Requires sufficient samples per category

---

## 11. References

**Key Papers:**
- Wang et al., "Text Embeddings by Weakly-Supervised Contrastive Pre-training" (E5, 2022)
- He et al., "DeBERTa: Decoding-enhanced BERT with Disentangled Attention" (2021)
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, 2021)
- Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision" (2023)
- Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (2017)

**Libraries & Tools:**
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- Sentence Transformers: https://www.sbert.net/
- OpenCLIP: https://github.com/mlfoundations/open_clip
- LightGBM Documentation: https://lightgbm.readthedocs.io/

---

**Document Version:** 1.1  
**Last Updated:** January 2026  
**Authors:** Vishesh Gupta  
**Competition:** Amazon ML Challenge 2025
