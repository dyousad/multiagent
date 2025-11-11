#!/usr/bin/env python3
"""Build full HotpotQA corpus with ALL samples and create embeddings cache."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cached_retrieval_manager import CachedRetrievalManager


def build_full_corpus():
    """Build corpus from ALL HotpotQA samples."""
    print("="*70)
    print("Building Full HotpotQA Corpus")
    print("="*70)
    print()

    data_path = "data/hotpot_dev_fullwiki_v1.json"
    output_path = "data/hotpotqa_corpus_full.json"

    print(f"Loading HotpotQA data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")
    print()

    corpus = []
    doc_id = 0

    print("Extracting documents from all samples...")
    for i, sample in enumerate(data):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(data)} samples...")

        context = sample.get('context', [])

        for title, sentences in context:
            if isinstance(sentences, list):
                for sent in sentences:
                    if sent.strip():
                        corpus.append({
                            "id": f"doc_{doc_id}",
                            "title": title,
                            "text": f"{title}: {sent}",
                            "sentence": sent
                        })
                        doc_id += 1
            else:
                if sentences.strip():
                    corpus.append({
                        "id": f"doc_{doc_id}",
                        "title": title,
                        "text": f"{title}: {sentences}",
                        "sentence": sentences
                    })
                    doc_id += 1

    print(f"\n✓ Extracted {len(corpus)} documents from {len(data)} samples")

    # Remove duplicates by text
    unique_texts = set()
    unique_corpus = []
    for doc in corpus:
        if doc["text"] not in unique_texts:
            unique_texts.add(doc["text"])
            unique_corpus.append(doc)

    print(f"✓ After deduplication: {len(unique_corpus)} unique documents")

    # Save corpus
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(unique_corpus, f, indent=2)

    print(f"✓ Corpus saved to {output_path}")
    print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    return output_path, len(unique_corpus)


def build_embeddings_cache(corpus_path: str):
    """Build and cache embeddings for the full corpus."""
    print()
    print("="*70)
    print("Building Embeddings Cache")
    print("="*70)
    print()

    print("This will:")
    print("  1. Load the full corpus")
    print("  2. Encode all documents using sentence transformers")
    print("  3. Build FAISS index")
    print("  4. Cache everything to disk")
    print()
    print("⏱️  This will take 5-15 minutes on first run")
    print("⚡ Subsequent runs will load from cache in seconds!")
    print()

    # Initialize retrieval manager (will build and cache)
    manager = CachedRetrievalManager(
        corpus_path=corpus_path,
        model_name="BAAI/bge-large-en-v1.5",
        cache_dir="data/cache",
        force_rebuild=True  # Force rebuild to create cache
    )

    print()
    print("="*70)
    print("✓ Cache Build Complete!")
    print("="*70)
    print()
    print(f"Cache location: data/cache/")
    print(f"Documents encoded: {len(manager.corpus)}")
    print(f"Embedding dimension: {manager.embeddings.shape[1]}")
    print()
    print("Next time the retrieval manager loads, it will use the cache")
    print("and initialize in seconds instead of minutes!")

    # Test retrieval
    print()
    print("Testing retrieval...")
    test_query = "What is Scott Derrickson's nationality?"
    results = manager.retrieve(test_query, top_k=3)
    print(f"\nQuery: {test_query}")
    print("Top 3 results:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc[:100]}...")


def main():
    """Main function."""
    print("="*70)
    print("HotpotQA Full Corpus Builder & Cache Generator")
    print("="*70)
    print()

    # Step 1: Build full corpus
    corpus_path, num_docs = build_full_corpus()

    # Step 2: Build embeddings cache
    build_embeddings_cache(corpus_path)

    print()
    print("="*70)
    print("✓ ALL DONE!")
    print("="*70)
    print()
    print("Summary:")
    print(f"  - Full corpus created: {corpus_path}")
    print(f"  - Total documents: {num_docs}")
    print(f"  - Embeddings cached: data/cache/")
    print()
    print("To use the cached retrieval in experiments:")
    print("  1. Set corpus_path='data/hotpotqa_corpus_full.json'")
    print("  2. The system will auto-load from cache")
    print()


if __name__ == "__main__":
    main()
