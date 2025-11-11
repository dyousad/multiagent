#!/usr/bin/env python3
"""Prepare corpus from HotpotQA data for retrieval."""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def extract_corpus_from_hotpotqa(
    data_path: str = "data/hotpot_dev_fullwiki_v1.json",
    output_path: str = "data/hotpotqa_corpus.json",
    max_samples: int = None
) -> List[Dict[str, Any]]:
    """Extract corpus from HotpotQA dataset.

    Parameters
    ----------
    data_path : str
        Path to HotpotQA data file.
    output_path : str
        Path to save the extracted corpus.
    max_samples : int
        Maximum number of samples to process (None for all).

    Returns
    -------
    List[Dict[str, Any]]
        Extracted corpus with text passages.
    """
    print(f"Loading HotpotQA data from {data_path}...")

    with open(data_path, 'r') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    print(f"Processing {len(data)} samples...")

    corpus = []
    doc_id = 0

    for sample in data:
        context = sample.get('context', [])

        # Each context is a list of [title, sentences] pairs
        for title, sentences in context:
            # Create a document for each paragraph
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
                # Single sentence case
                if sentences.strip():
                    corpus.append({
                        "id": f"doc_{doc_id}",
                        "title": title,
                        "text": f"{title}: {sentences}",
                        "sentence": sentences
                    })
                    doc_id += 1

    print(f"Extracted {len(corpus)} documents from corpus")

    # Save corpus
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(corpus, f, indent=2)

    print(f"Corpus saved to {output_path}")
    print(f"Sample document: {corpus[0]}")

    return corpus


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare HotpotQA corpus for retrieval")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/hotpot_dev_fullwiki_v1.json",
        help="Path to HotpotQA data file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/hotpotqa_corpus.json",
        help="Path to save corpus"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )

    args = parser.parse_args()

    try:
        corpus = extract_corpus_from_hotpotqa(
            data_path=args.data_path,
            output_path=args.output_path,
            max_samples=args.max_samples
        )
        print(f"\n✓ Corpus preparation complete!")
        print(f"  Total documents: {len(corpus)}")
        print(f"  Output: {args.output_path}")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
