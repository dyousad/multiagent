"""Retrieval manager for RAG-based evidence retrieval."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class RetrievalManager:
    """Manages retrieval of evidence passages using dense embeddings.

    This class:
    - Loads a corpus of documents
    - Encodes documents using sentence transformers
    - Builds a FAISS index for efficient similarity search
    - Retrieves top-k most relevant documents for a query
    """

    def __init__(
        self,
        corpus_path: str = "data/hotpotqa_corpus.json",
        model_name: str = "BAAI/bge-large-en-v1.5",
        top_k: int = 5,
        corpus: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize the retrieval manager.

        Parameters
        ----------
        corpus_path : str
            Path to the corpus JSON file.
        model_name : str
            Name of the sentence transformer model.
        top_k : int
            Default number of documents to retrieve.
        corpus : Optional[List[Dict[str, Any]]]
            Pre-loaded corpus (if provided, corpus_path is ignored).
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "sentence-transformers and faiss are required for RetrievalManager. "
                "Install them with: pip install sentence-transformers faiss-cpu"
            )

        self.model_name = model_name
        self.top_k = top_k

        # Load corpus
        if corpus is not None:
            self.corpus = corpus
        else:
            corpus_file = Path(corpus_path)
            if not corpus_file.exists():
                raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

            with open(corpus_file) as f:
                self.corpus = json.load(f)

        # Load sentence transformer model
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)

        # Extract text from corpus
        self.texts = [self._extract_text(doc) for doc in self.corpus]

        # Encode corpus
        print(f"Encoding {len(self.texts)} documents...")
        self.embeddings = self.model.encode(
            self.texts,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # Build FAISS index
        print("Building FAISS index...")
        embedding_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

        print(f"Retrieval manager initialized with {len(self.corpus)} documents")

    def _extract_text(self, doc: Dict[str, Any]) -> str:
        """Extract text from a document.

        Parameters
        ----------
        doc : Dict[str, Any]
            Document dictionary.

        Returns
        -------
        str
            Extracted text.
        """
        # Handle different corpus formats
        if "text" in doc:
            return doc["text"]
        elif "context" in doc:
            # HotpotQA format with title and sentences
            if isinstance(doc["context"], list):
                # List of [title, sentences] pairs
                texts = []
                for title, sentences in doc["context"]:
                    texts.append(f"{title}: {' '.join(sentences)}")
                return " ".join(texts)
            else:
                return str(doc["context"])
        elif "title" in doc and "sentences" in doc:
            return f"{doc['title']}: {' '.join(doc['sentences'])}"
        else:
            # Fall back to JSON string
            return json.dumps(doc)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Retrieve top-k most relevant documents for a query.

        Parameters
        ----------
        query : str
            The query string.
        top_k : Optional[int]
            Number of documents to retrieve (uses default if None).

        Returns
        -------
        List[str]
            List of retrieved document texts.
        """
        k = top_k or self.top_k

        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, k)

        # Return texts
        retrieved = [self.texts[idx] for idx in indices[0]]
        return retrieved

    def retrieve_with_scores(
        self, query: str, top_k: Optional[int] = None
    ) -> List[tuple[str, float]]:
        """Retrieve top-k documents with their similarity scores.

        Parameters
        ----------
        query : str
            The query string.
        top_k : Optional[int]
            Number of documents to retrieve.

        Returns
        -------
        List[tuple[str, float]]
            List of (document_text, score) tuples.
        """
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
