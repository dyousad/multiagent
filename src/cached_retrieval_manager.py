"""Enhanced retrieval manager with embedding caching."""

from __future__ import annotations

import json
import numpy as np
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class CachedRetrievalManager:
    """Retrieval manager with embedding and index caching.

    Key improvements:
    - Caches embeddings to disk to avoid re-encoding
    - Caches FAISS index for instant loading
    - Supports full corpus processing
    - Significantly faster initialization after first run
    """

    def __init__(
        self,
        corpus_path: str = "data/hotpotqa_corpus.json",
        model_name: str = "BAAI/bge-large-en-v1.5",
        top_k: int = 5,
        cache_dir: str = "data/cache",
        force_rebuild: bool = False,
        corpus: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize the cached retrieval manager.

        Parameters
        ----------
        corpus_path : str
            Path to the corpus JSON file.
        model_name : str
            Name of the sentence transformer model.
        top_k : int
            Default number of documents to retrieve.
        cache_dir : str
            Directory to store cached embeddings and index.
        force_rebuild : bool
            Force rebuild cache even if it exists.
        corpus : Optional[List[Dict[str, Any]]]
            Pre-loaded corpus (if provided, corpus_path is ignored).
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "sentence-transformers and faiss are required. "
                "Install: pip install sentence-transformers faiss-cpu"
            )

        self.model_name = model_name
        self.top_k = top_k
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate cache key based on corpus and model
        corpus_path_obj = Path(corpus_path)
        corpus_name = corpus_path_obj.stem
        model_clean = model_name.replace("/", "_")
        self.cache_key = f"{corpus_name}_{model_clean}"

        # Cache file paths
        self.embeddings_cache = self.cache_dir / f"{self.cache_key}_embeddings.npy"
        self.index_cache = self.cache_dir / f"{self.cache_key}_index.faiss"
        self.corpus_cache = self.cache_dir / f"{self.cache_key}_corpus.pkl"
        self.texts_cache = self.cache_dir / f"{self.cache_key}_texts.pkl"

        # Load corpus
        if corpus is not None:
            self.corpus = corpus
        else:
            if not corpus_path_obj.exists():
                raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

            with open(corpus_path_obj) as f:
                self.corpus = json.load(f)

        print(f"Corpus size: {len(self.corpus)} documents")

        # Check if cache exists and is valid
        cache_valid = (
            not force_rebuild
            and self.embeddings_cache.exists()
            and self.index_cache.exists()
            and self.corpus_cache.exists()
            and self.texts_cache.exists()
        )

        if cache_valid:
            print(f"Loading from cache: {self.cache_key}")
            self._load_from_cache()
        else:
            print(f"Building new embeddings and index...")
            self._build_and_cache()

        print(f"✓ Retrieval manager ready with {len(self.corpus)} documents")

    def _load_from_cache(self):
        """Load embeddings, index, and corpus from cache."""
        # Load sentence transformer model
        print(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        # Load cached data
        print("Loading embeddings from cache...")
        self.embeddings = np.load(self.embeddings_cache)

        print("Loading FAISS index from cache...")
        self.index = faiss.read_index(str(self.index_cache))

        print("Loading corpus metadata from cache...")
        with open(self.corpus_cache, 'rb') as f:
            self.corpus = pickle.load(f)

        with open(self.texts_cache, 'rb') as f:
            self.texts = pickle.load(f)

        print(f"✓ Loaded from cache in seconds (vs minutes for encoding)")

    def _build_and_cache(self):
        """Build embeddings and index, then cache them."""
        # Load model
        print(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        # Extract texts
        print("Extracting texts from corpus...")
        self.texts = [self._extract_text(doc) for doc in self.corpus]

        # Encode corpus
        print(f"Encoding {len(self.texts)} documents (this may take a while)...")
        self.embeddings = self.model.encode(
            self.texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32,  # Batch processing for efficiency
        )

        # Build FAISS index
        print("Building FAISS index...")
        embedding_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)

        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

        # Cache everything
        print("Caching embeddings and index for future use...")
        np.save(self.embeddings_cache, self.embeddings)
        faiss.write_index(self.index, str(self.index_cache))

        with open(self.corpus_cache, 'wb') as f:
            pickle.dump(self.corpus, f)

        with open(self.texts_cache, 'wb') as f:
            pickle.dump(self.texts, f)

        print(f"✓ Cache saved to {self.cache_dir}")

    def _extract_text(self, doc: Dict[str, Any]) -> str:
        """Extract text from a document."""
        if "text" in doc:
            return doc["text"]
        elif "context" in doc:
            if isinstance(doc["context"], list):
                texts = []
                for title, sentences in doc["context"]:
                    if isinstance(sentences, list):
                        texts.append(f"{title}: {' '.join(sentences)}")
                    else:
                        texts.append(f"{title}: {sentences}")
                return " ".join(texts)
            else:
                return str(doc["context"])
        elif "title" in doc and "sentence" in doc:
            return f"{doc['title']}: {doc['sentence']}"
        else:
            return json.dumps(doc)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Retrieve top-k most relevant documents."""
        k = top_k or self.top_k

        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, k)

        # Return texts
        retrieved = [self.texts[idx] for idx in indices[0]]
        return retrieved

    def retrieve_with_scores(
        self, query: str, top_k: Optional[int] = None
    ) -> List[tuple[str, float]]:
        """Retrieve top-k documents with scores."""
        k = top_k or self.top_k

        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, k)

        # Return texts with scores
        results = [
            (self.texts[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
        ]
        return results

    def clear_cache(self):
        """Clear all cached files."""
        for cache_file in [
            self.embeddings_cache,
            self.index_cache,
            self.corpus_cache,
            self.texts_cache,
        ]:
            if cache_file.exists():
                cache_file.unlink()
        print(f"✓ Cache cleared for {self.cache_key}")


# Backward compatibility: alias for original class
RetrievalManager = CachedRetrievalManager
