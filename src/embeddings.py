"""
Embedding generation for text and images.

Text encoders:
- E5-Large-v2 (retrieval-optimized)
- DeBERTa-v3-base (fine-tuned for price regression)

Image encoders:
- OpenCLIP ViT-L/14
- DINOv2-base (optional)
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from sklearn.decomposition import PCA
from joblib import dump as joblib_dump

# Text embeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Image embeddings
import open_clip


# ============================================================
# TEXT EMBEDDINGS: E5-LARGE-V2
# ============================================================

def combine_text_fields(name: str, desc: str) -> str:
    """Combine item_name and item_description for embedding."""
    name = str(name).strip() if pd.notna(name) else ""
    desc = str(desc).strip() if pd.notna(desc) else ""
    return (name + ". " + desc).strip()


def generate_e5_embeddings(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: Path,
    model_id: str = "intfloat/e5-large-v2",
    max_length: int = 512,
    batch_size: int = 32,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate E5-Large-v2 text embeddings (1024-dim, L2-normalized).
    
    Returns:
        X_train, X_val, X_test (numpy arrays)
    """
    print("\n" + "=" * 60)
    print("E5-LARGE-V2 TEXT EMBEDDINGS")
    print("=" * 60)
    
    # Prepare texts
    texts_train = [
        "passage: " + combine_text_fields(row.item_name, row.item_description)
        for _, row in df_train.iterrows()
    ]
    texts_val = [
        "passage: " + combine_text_fields(row.item_name, row.item_description)
        for _, row in df_val.iterrows()
    ]
    texts_test = [
        "passage: " + combine_text_fields(row.item_name, row.item_description)
        for _, row in df_test.iterrows()
    ]
    
    # Load model
    print(f"Loading model: {model_id}")
    model = SentenceTransformer(model_id, device=device)
    model.max_seq_length = max_length
    model.eval()
    
    # Encode
    @torch.no_grad()
    def encode_batch(texts):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            emb = model.encode(
                batch,
                batch_size=len(batch),
                device=device,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings.append(emb)
        return np.vstack(embeddings)
    
    print("Encoding train...")
    X_train = encode_batch(texts_train)
    print("Encoding val...")
    X_val = encode_batch(texts_val)
    print("Encoding test...")
    X_test = encode_batch(texts_test)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_text_tr_e5l2.npy", X_train)
    np.save(output_dir / "X_text_val_e5l2.npy", X_val)
    np.save(output_dir / "X_text_te_e5l2.npy", X_test)
    
    print(f"\nE5 embeddings saved:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape}")
    print(f"   Test:  {X_test.shape}")
    
    return X_train, X_val, X_test


# ============================================================
# TEXT EMBEDDINGS: DeBERTa (from fine-tuned model)
# ============================================================

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling with attention mask."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def generate_deberta_embeddings(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    model_dir: Path,
    output_dir: Path,
    max_length: int = 256,
    batch_size: int = 16,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract embeddings from fine-tuned DeBERTa-v3-base.
    
    Args:
        model_dir: Path to saved fine-tuned model checkpoint
    
    Returns:
        X_train, X_val, X_test (numpy arrays)
    """
    print("\n" + "=" * 60)
    print("DeBERTa-v3-base EMBEDDINGS (Fine-tuned)")
    print("=" * 60)
    
    # Prepare texts
    def prep_texts(df):
        return [combine_text_fields(row.item_name, row.item_description) 
                for _, row in df.iterrows()]
    
    texts_train = prep_texts(df_train)
    texts_val = prep_texts(df_val)
    texts_test = prep_texts(df_test)
    
    # Load model
    print(f"Loading from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), num_labels=1)
    model.to(device).eval()
    
    # Extract embeddings
    @torch.no_grad()
    def encode_batch(texts):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            tokens = tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=max_length, 
                return_tensors="pt"
            ).to(device)
            
            # Extract hidden states (not regression head)
            hidden = model.deberta(**tokens)[0]  # [batch, seq_len, 768]
            emb = mean_pool(hidden, tokens["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)  # L2 normalize
            embeddings.append(emb.cpu().numpy())
        
        return np.vstack(embeddings)
    
    print("Encoding train...")
    X_train = encode_batch(texts_train)
    print("Encoding val...")
    X_val = encode_batch(texts_val)
    print("Encoding test...")
    X_test = encode_batch(texts_test)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_text_train_deberta.npy", X_train)
    np.save(output_dir / "X_text_val_deberta.npy", X_val)
    np.save(output_dir / "X_text_test_deberta.npy", X_test)
    
    print(f"\nDeBERTa embeddings saved:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape}")
    print(f"   Test:  {X_test.shape}")
    
    return X_train, X_val, X_test


# ============================================================
# IMAGE EMBEDDINGS: OpenCLIP
# ============================================================

class ImageDataset(torch.utils.data.Dataset):
    """Dataset for loading cached images."""
    def __init__(self, df, cache_dir, preprocess):
        self.image_paths = [
            cache_dir / (link if link.endswith('.jpg') else f"{hash(link)}.jpg")
            for link in df['image_link']
        ]
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        if not path.exists():
            img = Image.new("RGB", (224, 224), (255, 255, 255))
        else:
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), (255, 255, 255))
        return self.preprocess(img)


def generate_clip_embeddings(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    image_cache_dir: Path,
    output_dir: Path,
    model_arch: str = "ViT-L-14",
    pretrained: str = "openai",
    batch_size: int = 192,
    num_workers: int = 8,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate OpenCLIP image embeddings (768-dim, L2-normalized).
    
    Returns:
        X_train, X_val, X_test (numpy arrays)
    """
    print("\n" + "=" * 60)
    print(f"OpenCLIP IMAGE EMBEDDINGS ({model_arch})")
    print("=" * 60)
    
    # Load model
    print(f"Loading {model_arch} ({pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_arch, 
        pretrained=pretrained
    )
    model = model.to(device).eval().half()
    
    # Create dataloaders
    def make_loader(df):
        dataset = ImageDataset(df, image_cache_dir, preprocess)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False
        )
    
    # Encode
    @torch.no_grad()
    def encode_split(loader, split_name):
        embeddings = []
        for batch in tqdm(loader, desc=f"Encoding {split_name}"):
            batch = batch.to(device, dtype=torch.float16)
            emb = model.encode_image(batch)
            emb = F.normalize(emb.float(), dim=-1)
            embeddings.append(emb.cpu().numpy())
        return np.vstack(embeddings)
    
    print("Encoding train...")
    X_train = encode_split(make_loader(df_train), "train")
    print("Encoding val...")
    X_val = encode_split(make_loader(df_val), "val")
    print("Encoding test...")
    X_test = encode_split(make_loader(df_test), "test")
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_img_tr_clip.npy", X_train)
    np.save(output_dir / "X_img_val_clip.npy", X_val)
    np.save(output_dir / "X_img_te_clip.npy", X_test)
    
    print(f"\nCLIP embeddings saved:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape}")
    print(f"   Test:  {X_test.shape}")
    
    return X_train, X_val, X_test


# ============================================================
# PCA DIMENSIONALITY REDUCTION
# ============================================================

def apply_pca_reduction(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 128,
    output_dir: Optional[Path] = None,
    name: str = "embedding"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    """
    Apply PCA reduction to embeddings (fit on train only).
    
    Returns:
        X_train_pca, X_val_pca, X_test_pca, pca_model
    """
    print(f"\nApplying PCA({n_components}) to {name}...")
    
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"   Explained variance: {explained_var:.4f}")
    print(f"   Reduced dims: {X_train.shape[1]} → {n_components}")
    
    # Save PCA model
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib_dump(pca, output_dir / f"pca_{name}.joblib")
    
    return X_train_pca, X_val_pca, X_test_pca, pca
