"""Neural agent to learn decomposer contribution from question embeddings and cost features.

This module provides a small PyTorch MLP that takes a question embedding and
scalar features (e.g. shapley_per_token, tokens_used, baseline) and predicts
an importance/contribution score. It also includes simple training and
prediction helpers.

The design is intentionally minimal and framework-agnostic: embeddings can be
provided by the caller (e.g. sentence-transformers). The module depends on
`torch` and `numpy` which are already listed in `requirements.txt`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # optional dependency; user may provide embeddings


class SimpleMLP(nn.Module):
    def __init__(self, emb_dim: int, num_scalar: int, hidden: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(emb_dim + num_scalar, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, emb: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb, scalars], dim=-1)
        return self.model(x).squeeze(-1)


class NeuralDecomposerAgent:
    """Wrapper around a small MLP for learning contribution scores.

    Parameters
    ----------
    emb_dim: int
        Dimensionality of question embeddings.
    num_scalar: int
        Number of scalar features concatenated with embedding (e.g. tokens, baseline).
    device: str
        Torch device, e.g. 'cpu' or 'cuda'.
    """

    def __init__(self, emb_dim: int = 384, num_scalar: int = 3, hidden: int = 256, device: Optional[str] = None):
        self.emb_dim = emb_dim
        self.num_scalar = num_scalar
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleMLP(emb_dim=emb_dim, num_scalar=num_scalar, hidden=hidden).to(self.device)

    @staticmethod
    def embed_text(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
        """Compute sentence embeddings using sentence-transformers if available.

        Returns a numpy array of shape (len(texts), emb_dim).
        """
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed; provide embeddings instead")
        model = SentenceTransformer(model_name)
        return np.asarray(model.encode(texts, convert_to_numpy=True))

    def predict(self, emb: np.ndarray, scalars: np.ndarray) -> np.ndarray:
        """Predict scores.

        emb: (N, emb_dim) or (emb_dim,)
        scalars: (N, num_scalar) or (num_scalar,)
        Returns: (N,) numpy array
        """
        self.model.eval()
        emb_t = torch.tensor(np.atleast_2d(emb).astype(np.float32), device=self.device)
        scal_t = torch.tensor(np.atleast_2d(scalars).astype(np.float32), device=self.device)
        with torch.no_grad():
            out = self.model(emb_t, scal_t)
        return out.cpu().numpy()

    def train(self, dataset: List[Dict[str, Any]], epochs: int = 10, batch_size: int = 32, lr: float = 1e-3):
        """Train the model.

        dataset: list of dicts with keys: 'embedding' (np.ndarray), 'scalars' (np.ndarray), 'target' (float)
        """
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        # build arrays
        X_emb = np.stack([d['embedding'] for d in dataset])
        X_scal = np.stack([d['scalars'] for d in dataset])
        y = np.asarray([d['target'] for d in dataset], dtype=np.float32)

        n = len(dataset)
        indices = np.arange(n)
        for epoch in range(epochs):
            np.random.shuffle(indices)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                batch_idx = indices[start:start + batch_size]
                emb_batch = torch.tensor(X_emb[batch_idx].astype(np.float32), device=self.device)
                scal_batch = torch.tensor(X_scal[batch_idx].astype(np.float32), device=self.device)
                y_batch = torch.tensor(y[batch_idx].astype(np.float32), device=self.device)

                preds = self.model(emb_batch, scal_batch)
                loss = loss_fn(preds, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(batch_idx)
            avg_loss = epoch_loss / max(1, n)
            print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.6f}")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'state_dict': self.model.state_dict(),
            'emb_dim': self.emb_dim,
            'num_scalar': self.num_scalar,
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data['state_dict'])


if __name__ == "__main__":
    # quick smoke test: train on random data
    print("NeuralDecomposerAgent smoke test")
    emb_dim = 384
    num_scalar = 3
    agent = NeuralDecomposerAgent(emb_dim=emb_dim, num_scalar=num_scalar)
    # create synthetic dataset: target correlates with sum of scalars
    N = 200
    X_emb = np.random.randn(N, emb_dim).astype(np.float32)
    X_scal = np.random.rand(N, num_scalar).astype(np.float32)
    y = (X_scal.sum(axis=1) + 0.1 * np.random.randn(N)).astype(np.float32)
    dataset = [{'embedding': X_emb[i], 'scalars': X_scal[i], 'target': float(y[i])} for i in range(N)]
    agent.train(dataset, epochs=3, batch_size=32, lr=1e-3)
    preds = agent.predict(X_emb[:5], X_scal[:5])
    print('preds:', preds)
