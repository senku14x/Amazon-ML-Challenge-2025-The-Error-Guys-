# **Amazon ML Challenge 2025: Smart Product Pricing Solution**

**Team:** The Error Guys 

**Member :** Ayush Sharma, Swapnil Saha, Souhardyo Dasgupta, Vishesh Gupta 

**Date:** October 13, 2025

## **Executive Summary**

We developed a semi-supervised multimodal ensemble combining text, image, and packaging features for e-commerce price prediction. Our solution evolved from an OptBlend baseline (37.7% SMAPE) to an enhanced architecture incorporating pseudo-labeling and cross-modal transformers, achieving significant improvements in generalization and stability.

## **Methodology**

### **Problem Analysis**

The challenge required predicting product prices from multimodal data (text descriptions, images, packaging metadata). Analysis revealed right-skewed price distributions motivating log-space regression, with strong correlations between description richness, pack quantity ratios, and pricing tiers.

### **Solution Architecture**

#### **Phase 1: Data Preparation & Embeddings**

* **Text Processing:** Parsed catalog content using regex patterns to extract item\_name, item\_description, and pack information  
* **Text Embeddings:**  
  * E5-Large-v2 (1024-dim → 128-dim PCA)  
  * DeBERTa-v3-base fine-tuned for price regression (768-dim → 128-dim PCA)  
* **Image Embeddings:**  
  * OpenCLIP ViT-L/14 (768-dim → 128-dim PCA)  
  * DINOv2-Base (768-dim → 128-dim PCA) \[Enhanced only\]  
* **Pack Features:** Extracted pack\_value, pack\_count, pack\_unit with canonicalization

  #### **Phase 2: Feature Engineering**

* **Cross-Modal Interactions:** Cosine similarities between modality pairs (E5-CLIP, DeBERTa-CLIP, DINO-all)  
* **Engineered Features:** Text complexity metrics, pack ratios (count×value, log(count/value)), missing indicators  
* **Dimensionality:** 433 total features before pruning

  #### **Phase 3: Model Training**

**Base Models (Heavily Regularized):**

* LightGBM: L1/L2 reg(25/50), num\_leaves=21, min\_data\_in\_leaf=600  
* XGBoost: max\_depth=5, min\_child\_weight=800, reg\_alpha=25  
* CatBoost: depth=5, l2\_leaf\_reg=50

**Cross-Modal Transformer \[Enhanced\]: ( Just for Testing the code has been removed. )**

* Architecture: 2 layers, 4 heads, d\_model=256  
* Token-wise modality embeddings with mean pooling

**Model Blending:**

* Baseline: Grid search optimal weights \- LGB(0.4) \+ XGB(0.3) \+ CAT(0.3)  
* Enhanced: Blend  :  LGB(0.4) \+ XGB(0.3) \+ CAT(0.3)  
* Both approaches showed stable performance with enhanced version prioritizing simplicity

  #### **Phase 4: Semi-Supervised Enhancement \[Enhanced Only\]**

1. **Pseudo-Labeling:** Generated labels for test set using best ensemble  
2. **Confidence Filtering:** Retained samples with std \< 0.05 (75k samples)  
3. **Weighted Training:** Combined 75k labeled (weight=1.0) \+ 75k pseudo (weight=0.5)  
4. **SHAP Pruning:** Selected top 191/433 features (80% cumulative importance)  
5. **Meta-Ensemble:** ElasticNetCV stacking of GBM.

   ## **Results & Validation**

   ### **Performance Metrics**

* **Baseline (OptBlend):** 37.7% SMAPE  
* **Enhanced Solution:** Improved generalization with tighter variance  
* **Validation MAE:** \~8.35 (price space)  
* **Cross-Model Correlation:** 0.994-0.999 (high stability)

  ### **Validation Suite**

1. **True-Labeled MAE:** Verified on out-of-fold predictions  
2. **Pseudo Consistency:** Confident predictions within 0.02 log-points  
3. **Distribution Sanity:** Aligned means (20.4 vs 23.6), no inflation  
4. **Feature Impact:** Cross-modal features contributed \~15% improvement

   ## **Code Submission**

   ### **Files Provided**

1. **`amazon_ml_challenge_optblend_baseline.py`**

   * Complete multimodal pipeline with CLIP, E5, DeBERTa  
   * Multi-GBM ensemble with optimal blending  
   * Establishes baseline performance  
2. **`amazon_ml_challenge_pseudo_final.py`**

   * All baseline components plus:  
   * DINOv2 embeddings, cross-modal transformer  
   * Weighted pseudo-labeling with SHAP pruning  
   * Ensemble using Blending of the pred.s

   ### **Key Innovations**

* **Weighted Pseudo-Labeling:** Leveraged unlabeled data with confidence weighting  
* **SHAP Feature Selection:** Data-driven dimensionality reduction  
* **Cross-Modal Transformer:** Captured non-linear modality interactions  
* **Heavy Regularization:** Prevented overfitting in high-dimensional space

  ### **Execution**

  \# Run baseline  
  python amazon\_ml\_challenge\_optblend\_baseline.py  
  \# Run enhanced solution  
  python amazon\_ml\_challenge\_pseudo\_final.py


  ## **Technical Details**

**Environment:** Google Colab Pro (A100 GPU), PyTorch 2.x, Transformers 4.44+  
 **Data Split:** 80/20 stratified by log-price quantiles  
 **Training Time:** \~4 hours total  
 **Storage:** `/content/drive/MyDrive/smart_product_pricing_final/`

## **Conclusion**

Our solution demonstrates effective integration of multimodal representations with semi-supervised learning. The combination of confidence-weighted pseudo-labeling and SHAP-based feature selection proved particularly effective in reducing noise while maintaining prediction accuracy, ensuring robust performance across diverse product categories.

