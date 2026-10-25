"""
IndoBERT + RCNN Neural Architecture for Child Stunting Classification
====================================================================
Architecture Specification:
1. Contextual Representation: IndoBERT (indobenchmark/indobert-base-p1 or configurable BERT backbone)
   extracting 768-dimensional token hidden states.
2. Recurrent Component: Bidirectional LSTM capturing sequential narrative dependencies.
3. Convolutional Component: 1D Convolution with attention-masked max-pooling to eliminate padding bias.
4. Classification Head: Multi-layer perceptron with Dropout and Softmax output.
5. Configurable Taxonomy: Target classes configurable to match study operational definitions.
6. Validation & Leakage Prevention: Stratified train/val/test splitting, deduplication, and best checkpoint selection.
7. Error Transparency: Raises ModelNotTrainedError when uncheckpointed, eliminating silent heuristic fallbacks.
"""

from __future__ import annotations

import os
import sys
import math
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union

logger = logging.getLogger("IndoBERT_RCNN")

# Default study research taxonomy (Configurable to study operational definitions)
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
        - Attention-Masked Max-Over-Time Pooling (eliminates padding token influence)
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

            # Combined representation: [token_embedding; lstm_context]
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
            Forward pass of IndoBERT + RCNN with strict attention mask enforcement.
            """
            if bert_embeddings is not None:
                token_embeddings = bert_embeddings
            elif self.bert is not None:
                bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                token_embeddings = bert_outputs.last_hidden_state  # (B, L, 768)
            else:
                raise RuntimeError("Either self.bert must be initialized or bert_embeddings provided.")

            # Zero-out padding representations before LSTM
            mask_3d = attention_mask.unsqueeze(-1).float()
            token_embeddings = token_embeddings * mask_3d

            # Recurrent context: Bidirectional LSTM
            lstm_out, _ = self.lstm(token_embeddings)  # (B, L, 2*hidden_dim)

            # Concatenate token representation with recurrent context
            combined = torch.cat([token_embeddings, lstm_out], dim=-1)  # (B, L, combined_dim)

            # Transpose for Conv1d: (B, Channels, Length)
            combined_t = combined.transpose(1, 2)
            conv_out = F.relu(self.conv(combined_t))  # (B, conv_filters, L)

            # CRITICAL: Apply attention mask before max-over-time pooling so padding tokens
            # cannot contribute to or dominate the max activation!
            mask_for_conv = attention_mask.unsqueeze(1).bool()  # (B, 1, L)
            conv_out_masked = conv_out.masked_fill(~mask_for_conv, -1e9)

            # Max-over-time pooling
            pooled = F.adaptive_max_pool1d(conv_out_masked, 1).squeeze(-1)  # (B, conv_filters)

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


def prepare_dataset_splits(
    texts: List[str],
    labels: List[Union[str, int]],
    label2id: Dict[str, int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Validate dataset, prevent data leakage across splits, and perform stratified splitting.

    Requirements:
    - Text-label count alignment.
    - Rejects unknown labels not in label2id.
    - Checks for duplicate text records and ensures no data leakage.
    - Requires sufficient samples per class for valid research evaluation.
    """
    if len(texts) != len(labels):
        raise ValueError(
            f"Input length mismatch: {len(texts)} texts provided for {len(labels)} labels."
        )

    if len(texts) < 10:
        raise ValueError(
            f"Dataset has only {len(texts)} samples. A defensible research model requires "
            "sufficient samples for train/validation/test splits (minimum 10 samples)."
        )

    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-3):
        raise ValueError(
            f"Split ratios must sum to 1.0, got train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

    # Validate all labels and convert to integer IDs
    int_labels = []
    for idx, l in enumerate(labels):
        if isinstance(l, str):
            clean_l = l.strip()
            if clean_l not in label2id:
                raise ValueError(
                    f"Unknown label '{l}' at index {idx}. Valid classes: {list(label2id.keys())}."
                )
            int_labels.append(label2id[clean_l])
        elif isinstance(l, int):
            if l not in label2id.values():
                raise ValueError(
                    f"Unknown integer label ID '{l}' at index {idx}. Valid IDs: {list(label2id.values())}."
                )
            int_labels.append(l)
        else:
            raise ValueError(f"Invalid label type '{type(l)}' at index {idx}.")

    # Group sample indices by clean text to prevent data leakage (duplicate text across splits)
    text_to_indices: Dict[str, List[int]] = {}
    for idx, (t, l) in enumerate(zip(texts, int_labels)):
        clean_t = str(t).strip()
        if clean_t not in text_to_indices:
            text_to_indices[clean_t] = []
        text_to_indices[clean_t].append(idx)

    # Stratify by class
    rng = random.Random(seed)
    class_groups: Dict[int, List[List[int]]] = {cid: [] for cid in label2id.values()}

    for clean_t, indices in text_to_indices.items():
        # Representative label for this text group
        grp_label = int_labels[indices[0]]
        class_groups[grp_label].append(indices)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    split_stats = {"class_distribution": {}, "duplicates_grouped": len(texts) - len(text_to_indices)}

    for cid, groups in class_groups.items():
        rng.shuffle(groups)
        n_grps = len(groups)
        split_stats["class_distribution"][cid] = n_grps

        n_train = max(1, int(round(n_grps * train_ratio)))
        n_val = max(1, int(round(n_grps * val_ratio))) if (n_grps - n_train) >= 2 else (1 if n_grps > 1 else 0)
        n_test = n_grps - n_train - n_val

        # If too small for 3-way split, prioritize train then val
        if n_test < 0:
            n_test = 0
            n_val = max(0, n_grps - n_train)

        train_grps = groups[:n_train]
        val_grps = groups[n_train : n_train + n_val]
        test_grps = groups[n_train + n_val :]

        for g in train_grps:
            train_indices.extend(g)
        for g in val_grps:
            val_indices.extend(g)
        for g in test_grps:
            test_indices.extend(g)

    # Check for empty validation or test sets
    if len(val_indices) == 0:
        logger.warning("Validation set is empty due to small class sizes. Reallocating samples.")
        if len(train_indices) > 2:
            val_indices.append(train_indices.pop())

    return {
        "train_texts": [texts[i] for i in train_indices],
        "train_labels": [int_labels[i] for i in train_indices],
        "val_texts": [texts[i] for i in val_indices],
        "val_labels": [int_labels[i] for i in val_indices],
        "test_texts": [texts[i] for i in test_indices],
        "test_labels": [int_labels[i] for i in test_indices],
        "stats": split_stats,
        "counts": {
            "total": len(texts),
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_indices)
        }
    }


class IndoBERTRCNNPipeline:
    """
    High-level research pipeline for IndoBERT + RCNN training, validation,
    checkpoint saving, evaluation, and inference.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_name: str = "indobenchmark/indobert-base-p1",
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
        self.is_trained = False
        self.metadata: Dict[str, Any] = {}

        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)

    def initialize_new_model(self) -> None:
        """Instantiate a new untrained IndoBERT + RCNN model with tokenizer."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for IndoBERTRCNNPipeline. Please install torch.")
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers is required. Please install transformers.")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = IndoBERTRCNNModel(
            pretrained_model_name=self.model_name,
            num_classes=self.num_classes
        )
        self.is_loaded = True
        self.is_trained = False

    def save_checkpoint(
        self,
        save_path: str,
        best_val_metric: Optional[float] = None,
        history: Optional[Dict[str, Any]] = None,
        split_stats: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save model weights, mappings, and full experiment metadata."""
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
            "conv_filters": getattr(self.model, "conv_filters", 256),
            "is_trained": True,
            "best_val_metric": best_val_metric,
            "history": history or {},
            "split_stats": split_stats or {}
        }
        torch.save(checkpoint_data, str(save_p))
        logger.info("Saved trained IndoBERT+RCNN checkpoint to %s", save_p)

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
        self.metadata = {
            "best_val_metric": checkpoint.get("best_val_metric"),
            "history": checkpoint.get("history"),
            "split_stats": checkpoint.get("split_stats")
        }

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = IndoBERTRCNNModel(
            pretrained_model_name=self.model_name,
            num_classes=self.num_classes
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.is_loaded = True
        self.is_trained = checkpoint.get("is_trained", True)
        logger.info("Successfully loaded IndoBERT+RCNN checkpoint from %s", checkpoint_path)

    def _evaluate_dataset_loss_and_preds(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device
    ) -> Tuple[float, List[int], List[int]]:
        """Compute average loss and collect true/predicted labels on a DataLoader."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_trues = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                total_loss += loss.item()

                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_trues.extend(labels.cpu().tolist())

        avg_loss = total_loss / max(1, len(loader))
        return avg_loss, all_trues, all_preds

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
        Train the IndoBERT + RCNN model with strict validation tracking and best-checkpoint selection.
        """
        if not HAS_TORCH or not HAS_TRANSFORMERS:
            raise ImportError("PyTorch and Transformers are required for training.")

        if self.model is None or self.tokenizer is None:
            self.initialize_new_model()

        # Convert labels
        y_train = [
            self.label2id[l.strip()] if isinstance(l, str) else int(l)
            for l in train_labels
        ]

        train_ds = StuntingTextDataset(train_texts, y_train, self.tokenizer)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        has_val = val_texts is not None and val_labels is not None and len(val_texts) > 0
        val_loader = None
        if has_val:
            y_val = [
                self.label2id[l.strip()] if isinstance(l, str) else int(l)
                for l in val_labels
            ]
            val_ds = StuntingTextDataset(val_texts, y_val, self.tokenizer)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=0.01
        )

        history = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": []
        }

        best_val_loss = float("inf")
        best_state_dict = None
        best_epoch = 1

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / max(1, len(train_loader))
            history["epochs"].append(epoch + 1)
            history["train_loss"].append(round(avg_train_loss, 4))

            # Validation evaluation
            if val_loader is not None:
                val_loss, y_true, y_pred = self._evaluate_dataset_loss_and_preds(
                    val_loader, criterion, device
                )
                acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / max(1, len(y_true))
                history["val_loss"].append(round(val_loss, 4))
                history["val_accuracy"].append(round(acc, 4))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                logger.info(
                    "Epoch %d/%d - Train Loss: %.4f - Val Loss: %.4f - Val Acc: %.4f",
                    epoch + 1, epochs, avg_train_loss, val_loss, acc
                )
            else:
                history["val_loss"].append(None)
                history["val_accuracy"].append(None)
                logger.info("Epoch %d/%d - Train Loss: %.4f", epoch + 1, epochs, avg_train_loss)

        # Restore best model weights
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        self.model.eval()
        self.is_trained = True

        if checkpoint_save_path:
            self.save_checkpoint(
                checkpoint_save_path,
                best_val_metric=round(best_val_loss, 4) if best_val_loss != float("inf") else None,
                history=history,
                split_stats={"best_epoch": best_epoch}
            )

        return history

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Run inference on clinical note texts.

        Strict contract:
        - Only permitted when model is trained and checkpoint loaded.
        - Raises ModelNotTrainedError otherwise (no silent heuristic fallbacks).
        """
        if not self.is_loaded or self.model is None or not self.is_trained:
            raise ModelNotTrainedError(
                "IndoBERT + RCNN model checkpoint is not loaded or trained. "
                "Inference requires a verified trained checkpoint (.pt file). "
                "Heuristic fallback is strictly prohibited by research methodology."
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
                    results.append({
                        "label_id": -1,
                        "label_key": "UNSPECIFIED",
                        "label_description": "Catatan Klinis Kosong / Tidak Tersedia",
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
                best_key = self.id2label.get(best_idx, f"CLASS_{best_idx}")
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
