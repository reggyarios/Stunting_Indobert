"""
IndoBERT + RCNN Neural Architecture for Child Stunting Classification
====================================================================
Architecture Specification:
1. Contextual Representation: IndoBERT (indobenchmark/indobert-base-p1 or bert-base-multilingual-cased)
   extracting 768-dimensional token hidden states.
2. Recurrent Component: Bidirectional LSTM / GRU capturing forward and backward clinical narrative dependencies.
3. Convolutional Component: 1D Convolution with adaptive max-pooling to extract salient local n-gram clinical features.
4. Classification Head: Multi-layer perceptron with Dropout and Softmax output.
5. Checkpoint Persistence: Full state_dict, label mapping, and hyperparameter checkpointing.
6. Error Transparency: Raises ModelNotTrainedError when uncheckpointed, eliminating silent heuristic fallbacks.
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union

logger = logging.getLogger("IndoBERT_RCNN")

# Canonical clinical stunting taxonomy (Permenkes No. 2/2020 & WHO 2006)
DEFAULT_LABEL2ID: Dict[str, int] = {
    "STUNTING_BERAT_KOMORBID": 0,
    "STUNTING_KRONIS": 1,
    "STUNTING_SEDANG": 2,
    "GIZI_BURUK_AKUT": 3,
    "GIZI_KURANG_RISIKO": 4,
    "NORMAL_OPTIMAL": 5,
    "RISIKO_GIZI_LEBIH": 6
}

DEFAULT_ID2LABEL: Dict[int, str] = {v: k for k, v in DEFAULT_LABEL2ID.items()}

DEFAULT_LABEL_DESCRIPTIONS: Dict[str, str] = {
    "STUNTING_BERAT_KOMORBID": "Stunting Berat dengan Komorbiditas Klinis",
    "STUNTING_KRONIS": "Stunting Kronis / Sangat Pendek (HAZ < -3 SD)",
    "STUNTING_SEDANG": "Stunting Sedang / Pendek (-3 SD <= HAZ < -2 SD)",
    "GIZI_BURUK_AKUT": "Gizi Buruk Akut / Wasting Berat (WHZ < -3 SD)",
    "GIZI_KURANG_RISIKO": "Gizi Kurang / Risiko Gagal Tumbuh",
    "NORMAL_OPTIMAL": "Pertumbuhan Normal / Gizi Optimal",
    "RISIKO_GIZI_LEBIH": "Risiko Gizi Lebih / Overweight"
}


class ModelNotTrainedError(Exception):
    """Raised when inference is attempted without a valid trained model checkpoint."""
    pass


# Check availability of PyTorch and Transformers
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None
    Dataset = object
    DataLoader = None

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoTokenizer = None
    AutoModel = None


if HAS_TORCH:
    class IndoBERTRCNNModel(nn.Module):
        """
        PyTorch IndoBERT + RCNN Neural Network Architecture.

        Layers:
        - IndoBERT Transformer Encoder (768 dim)
        - Bidirectional LSTM (hidden_dim=256 -> 512 total bidirectional)
        - 1D Convolution (in_channels=768+512=1280, out_channels=256, kernel_size=3, padding=1)
        - Max-Over-Time Pooling
        - Dense Classification Head: Linear(256, 128) -> ReLU -> Dropout(0.3) -> Linear(128, num_classes)
        """

        def __init__(
            self,
            pretrained_model_name: str = "indobenchmark/indobert-base-p1",
            hidden_dim: int = 256,
            conv_filters: int = 256,
            kernel_size: int = 3,
            num_classes: int = 7,
            dropout: float = 0.3,
            freeze_bert_layers: int = 8
        ):
            super().__init__()
            self.pretrained_model_name = pretrained_model_name
            self.hidden_dim = hidden_dim
            self.conv_filters = conv_filters
            self.num_classes = num_classes

            # IndoBERT backbone
            if HAS_TRANSFORMERS and AutoModel is not None:
                self.bert = AutoModel.from_pretrained(pretrained_model_name)
                bert_dim = self.bert.config.hidden_size
                # Optionally freeze lower BERT layers for efficient training
                if freeze_bert_layers > 0 and hasattr(self.bert, "encoder"):
                    for layer in self.bert.encoder.layer[:freeze_bert_layers]:
                        for param in layer.parameters():
                            param.requires_grad = False
            else:
                bert_dim = 768
                self.bert = None

            # Recurrent component: Bidirectional LSTM
            self.lstm = nn.LSTM(
                input_size=bert_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                bidirectional=True,
                batch_first=True
            )

            # Combined representation: [left_context; word_embedding; right_context]
            combined_dim = bert_dim + (2 * hidden_dim)

            # Convolutional component: 1D CNN over combined representations
            self.conv = nn.Conv1d(
                in_channels=combined_dim,
                out_channels=conv_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )

            # Classification Head
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(conv_filters, 128)
            self.relu = nn.ReLU()
            self.fc_out = nn.Linear(128, num_classes)

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            bert_embeddings: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Forward pass of IndoBERT + RCNN.

            Parameters
            ----------
            input_ids : Tensor of shape (batch_size, seq_len)
            attention_mask : Tensor of shape (batch_size, seq_len)
            bert_embeddings : Optional precomputed embeddings (batch_size, seq_len, 768)

            Returns
            -------
            logits : Tensor of shape (batch_size, num_classes)
            """
            if bert_embeddings is not None:
                token_embeddings = bert_embeddings
            elif self.bert is not None:
                bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                token_embeddings = bert_outputs.last_hidden_state  # (B, L, 768)
            else:
                raise RuntimeError("Either self.bert must be initialized or bert_embeddings provided.")

            # Recurrent context: Bidirectional LSTM
            lstm_out, _ = self.lstm(token_embeddings)  # (B, L, 2*hidden_dim)

            # Concatenate token representation with recurrent context
            combined = torch.cat([token_embeddings, lstm_out], dim=-1)  # (B, L, 768 + 2*hidden_dim)

            # Transpose for Conv1d: (B, Channels, Length)
            combined_t = combined.transpose(1, 2)
            conv_out = F.relu(self.conv(combined_t))  # (B, conv_filters, L)

            # Max-over-time pooling
            pooled = F.adaptive_max_pool1d(conv_out, 1).squeeze(-1)  # (B, conv_filters)

            # Dense Classification
            x = self.dropout(pooled)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            logits = self.fc_out(x)  # (B, num_classes)
            return logits


class StuntingTextDataset(Dataset):
    """PyTorch Dataset for clinical stunting texts."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item


class IndoBERTRCNNPipeline:
    """
    High-level research pipeline for IndoBERT + RCNN training, inference,
    checkpoint saving, and validation.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_name: str = "indobenchmark/indobert-base-p1",
        num_classes: int = 7,
        label2id: Optional[Dict[str, int]] = None
    ):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.label2id = label2id or DEFAULT_LABEL2ID.copy()
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_classes = len(self.label2id)

        self.model = None
        self.tokenizer = None
        self.is_loaded = False

        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)

    def initialize_new_model(self) -> None:
        """Instantiate a new untrained IndoBERT + RCNN model with tokenizer."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for IndoBERTRCNNPipeline. Please install torch: pip install torch")
        if not HAS_TRANSFORMERS:
            raise ImportError("HuggingFace Transformers is required. Please install: pip install transformers")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = IndoBERTRCNNModel(
            pretrained_model_name=self.model_name,
            num_classes=self.num_classes
        )
        self.is_loaded = True

    def save_checkpoint(self, save_path: str) -> None:
        """Save model state dict, labels, and metadata to .pt file."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Cannot save an uninitialized model.")

        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_data = {
            "model_state_dict": self.model.state_dict(),
            "model_name": self.model_name,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "num_classes": self.num_classes,
            "hidden_dim": getattr(self.model, "hidden_dim", 256),
            "conv_filters": getattr(self.model, "conv_filters", 256)
        }
        torch.save(checkpoint_data, str(save_p))
        logger.info("Saved IndoBERT+RCNN checkpoint to %s", save_p)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load trained model weights and mappings from checkpoint file."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required to load checkpoint.")
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers is required to load tokenizer.")

        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        checkpoint = torch.load(str(p), map_location=torch.device("cpu"))
        self.label2id = checkpoint.get("label2id", self.label2id)
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_classes = len(self.label2id)
        self.model_name = checkpoint.get("model_name", self.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = IndoBERTRCNNModel(
            pretrained_model_name=self.model_name,
            num_classes=self.num_classes
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.is_loaded = True
        logger.info("Successfully loaded IndoBERT+RCNN checkpoint from %s", checkpoint_path)

    def train(
        self,
        train_texts: List[str],
        train_labels: List[Union[str, int]],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[Union[str, int]]] = None,
        epochs: int = 3,
        batch_size: int = 8,
        lr: float = 2e-5,
        checkpoint_save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the IndoBERT + RCNN model on labeled clinical stunting records.
        """
        if not HAS_TORCH or not HAS_TRANSFORMERS:
            raise ImportError("PyTorch and Transformers are required for training.")

        if self.model is None or self.tokenizer is None:
            self.initialize_new_model()

        # Convert string labels to integer IDs if needed
        y_train = [
            self.label2id[l] if isinstance(l, str) else int(l)
            for l in train_labels
        ]

        train_ds = StuntingTextDataset(train_texts, y_train, self.tokenizer)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=0.01
        )

        history = {"epochs": [], "loss": [], "val_accuracy": []}

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(train_loader))
            history["epochs"].append(epoch + 1)
            history["loss"].append(round(avg_loss, 4))
            logger.info("Epoch %d/%d - Loss: %.4f", epoch + 1, epochs, avg_loss)

        self.model.eval()
        if checkpoint_save_path:
            self.save_checkpoint(checkpoint_save_path)

        return history

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of clinical note texts.

        Returns
        -------
        List of dicts:
        {
            "label_id": int,
            "label_key": str,
            "label_description": str,
            "confidence": float,
            "probabilities": Dict[str, float]
        }
        """
        if not self.is_loaded or self.model is None:
            raise ModelNotTrainedError(
                "IndoBERT + RCNN model checkpoint is not loaded or trained. "
                "Inference requires a trained checkpoint (.pt file) with actual weights. "
                "Heuristic fallback is strictly disabled in accordance with the research methodology."
            )

        if not texts:
            return []

        cleaned_texts = [str(t).strip() if t is not None else "" for t in texts]
        results: List[Dict[str, Any]] = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            for t in cleaned_texts:
                if not t:
                    # Safe handling for empty text
                    results.append({
                        "label_id": -1,
                        "label_key": "UNSPECIFIED",
                        "label_description": "Teks Catatan Klinis Kosong / Tidak Tersedia",
                        "confidence": 0.0,
                        "probabilities": {}
                    })
                    continue

                encoded = self.tokenizer(
                    t,
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                    return_tensors="pt"
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                logits = self.model(input_ids, attention_mask)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

                best_idx = int(probs.argmax())
                best_conf = float(probs[best_idx])
                best_key = self.id2label.get(best_idx, "UNKNOWN")
                best_desc = DEFAULT_LABEL_DESCRIPTIONS.get(best_key, best_key)

                prob_dict = {
                    self.id2label.get(i, f"CLASS_{i}"): round(float(p), 4)
                    for i, p in enumerate(probs)
                }

                results.append({
                    "label_id": best_idx,
                    "label_key": best_key,
                    "label_description": best_desc,
                    "confidence": round(best_conf, 4),
                    "probabilities": prob_dict
                })

        return results
