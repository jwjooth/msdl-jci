import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Computes attention-weighted mean pooling over token embeddings.

    Formula:
        mean_embedding = sum(token_embeddings * attention_mask) / sum(attention_mask)

    Args:
        model_output: HuggingFace model output containing `last_hidden_state`.
        attention_mask: Tensor of shape (batch_size, seq_len) indicating non-padded tokens.

    Returns:
        Tensor of shape (batch_size, hidden_dim) containing sentence-level embeddings.
    """
    token_embeddings = model_output.last_hidden_state  # Shape: [batch_size, seq_len, 768]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )  # Shape: [batch_size, seq_len, 768]

    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def extract_indobert_embeddings(
    input_csv: str = "processed_daily_news.csv",
    output_csv: str = "daily_news_embeddings.csv",
    model_name: str = "indobenchmark/indobert-base-p2",
    batch_size: int = 16,
    max_length: int = 512,
    device: str = None,
) -> pd.DataFrame:
    """
    Extracts dense 768-dimensional embeddings for daily news texts using frozen IndoBERT.
    """
    # 1. Device Configuration
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    print(f"[INFO] Using device: {device.upper()}")

    # 2. Load Input Dataset
    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            f"Input file '{input_csv}' not found. Please run preprocessing first."
        )

    df = pd.read_csv(input_csv)
    print(f"[INFO] Loaded {len(df)} daily records from '{input_csv}'")

    # Combine title & content for richer contextual signals
    texts = (df["title"].fillna("") + ". " + df["content"].fillna("")).str.strip().tolist()

    # 3. Model & Tokenizer Initialization
    print(f"[INFO] Initializing tokenizer & frozen model: '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # 4. Explicitly Freeze Model Weights & Set to Eval Mode
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.to(device_obj)

    # 5. Batched Inference Loop
    all_embeddings = []
    total_batches = int(np.ceil(len(texts) / batch_size))

    print(
        f"[INFO] Starting text feature extraction (batch_size={batch_size}, max_length={max_length})..."
    )
    with torch.no_grad():
        for i in tqdm(
            range(0, len(texts), batch_size), total=total_batches, desc="Extracting Embeddings"
        ):
            batch_texts = texts[i : i + batch_size]

            # Tokenization with padding and truncation
            encoded_inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device_obj)

            # Forward pass through frozen IndoBERT
            outputs = model(**encoded_inputs)

            # Attention-aware Mean Pooling
            pooled_embeddings = mean_pooling(outputs, encoded_inputs["attention_mask"])

            # Transfer to CPU numpy array
            all_embeddings.append(pooled_embeddings.cpu().numpy())

    # 6. Aggregate & Map to DataFrame
    embedding_matrix = np.vstack(all_embeddings)  # Shape: [N, 768]
    hidden_dim = embedding_matrix.shape[1]
    emb_cols = [f"emb_{idx}" for idx in range(hidden_dim)]

    emb_df = pd.DataFrame(embedding_matrix, columns=emb_cols)
    emb_df.insert(0, "trade_date", df["trade_date"].values)

    # 7. Save to CSV
    emb_df.to_csv(output_csv, index=False)
    print(
        f"[SUCCESS] Dense text feature representations ({len(emb_df)}x{hidden_dim}) saved to '{output_csv}'!"
    )

    return emb_df


if __name__ == "__main__":
    extract_indobert_embeddings(
        input_csv="processed_daily_news.csv",
        output_csv="daily_news_embeddings.csv",
        model_name="indobenchmark/indobert-base-p2",
        batch_size=16,
        max_length=512,
    )
