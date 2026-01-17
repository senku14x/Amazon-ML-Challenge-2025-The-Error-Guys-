# Amazon ML Challenge 2025 - Smart Product Pricing

**Top 150 / 21,000 Teams**

Multimodal deep learning system for predicting e-commerce product prices from catalog descriptions and product images.

---

## Problem Statement

Given product catalog text and images, predict the retail price across diverse categories (electronics, groceries, apparel, etc.) while handling:
- Missing or incomplete data
- Multi-language descriptions  
- Wide price range variance ($1 - $10,000+)

---

## Approach

### Multimodal Feature Fusion
- **Text Embeddings:** E5-Large-v2 (1024-dim) + fine-tuned DeBERTa-v3-base (768-dim)
- **Image Embeddings:** OpenCLIP ViT-L/14 (768-dim) + DINOv2-base (768-dim)
- **Structured Features:** Pack quantity, value extraction, text complexity metrics

### Cross-Modal Intelligence
Created interaction features between modalities:
- Cosine similarity (text ↔ image alignment)
- Norm ratios and magnitude differences
- Pack-size × semantic-similarity interactions

**Impact:** +15% improvement over baseline embeddings alone

### Robust Ensemble
- **Base Models:** LightGBM, XGBoost, CatBoost (heavily regularized)
- **Stacking:** Weighted blending optimized via grid search on validation SMAPE
- **Regularization:** L1=25, L2=50 to prevent overfitting on 75k samples

---

## Results

| Metric | Score |
|--------|-------|
| **Validation SMAPE** | 37.7% → 35.2% (optimized) |
| **Leaderboard Rank** | **Top 150 / 21,000** |
| **Cross-Model Correlation** | 0.994+ (high stability) |

---

## Tech Stack

**Core ML:** PyTorch • Transformers (HuggingFace) • sentence-transformers • OpenCLIP

**Gradient Boosting:** LightGBM • XGBoost • CatBoost

**Infrastructure:** Google Colab Pro (A100 GPU) • scikit-learn • pandas • numpy

---

## Repository Structure
```
├── src/
│   ├── preprocessing.py      # Data cleaning & parsing
│   ├── embeddings.py          # Text/image encoding
│   ├── feature_engineering.py # Cross-modal features
│   └── modeling.py            # GBM training & ensemble
├── docs/
│   ├── METHODOLOGY.md         # Detailed technical approach
│   └── architecture.png       # System diagram
├── notebooks/
│   └── exploration.ipynb      # EDA & experiments
└── README.md
```

---

## Key Innovations

1. **Dual Text Encoders**  
   Combined E5 (retrieval-optimized) + task-specific fine-tuned DeBERTa for richer semantic capture

2. **Cross-Modal Validation**  
   Measured text-image alignment via cosine similarity to detect mismatched listings

3. **Pseudo-Labeling Strategy**  
   High-confidence test predictions (σ < 0.05) added to training with 0.5× weight → +2% gain

4. **Feature Pruning**  
   SHAP-based selection retained top 80% important features (433 → 191), reducing noise

---

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (preprocessing → embeddings → training)
python src/main.py --config config.yaml

# Generate submission
python src/predict.py --model_dir models/ --output submission.csv
```

---

## Model Performance Breakdown

| Component | Validation MAE | SMAPE |
|-----------|---------------|-------|
| E5 + CLIP baseline | 8.89 | 39.1% |
| + DeBERTa + DINOv2 | 8.52 | 37.8% |
| + Cross-modal features | 8.35 | 36.4% |
| + Pseudo-labeling | 8.21 | 35.2% |

---

## Learnings & Challenges

**What Worked:**
- Heavy regularization (prevented overfitting on limited data)
- Cross-modal features captured listing quality signals
- Conservative pseudo-labeling (high threshold avoided noise)

**What Didn't:**
- Deep fusion networks (overfitted, worse than linear blending)
- External data augmentation (out-of-distribution hurt more than helped)
- Price regime clustering (added complexity without gains)

---

## Citation
```bibtex
@misc{gupta2025amazonml,
  author = {Gupta, Vishesh and Team},
  title = {Smart Product Pricing via Multimodal Deep Learning},
  year = {2025},
  publisher = {Amazon ML Challenge},
  url = {https://github.com/visheshgupta/amazon-ml-challenge}
}
```

---

## Team

**The Error Guys**  
Vishesh Gupta • Ayush Sharma • Swapnil Saha • Souhardyo Dasgupta

---

## Contact

**Vishesh Gupta**  
Electrical Engineering  • Jadavpur University  
Email: visheshguptaw14x@gmail.com.com • LinkedIn: [linkedin.com/in/yourprofile](https://www.linkedin.com/in/vishesh-gupta-33927932b/) • 

---

