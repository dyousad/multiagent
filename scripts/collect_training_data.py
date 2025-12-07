#!/usr/bin/env python3
"""Collect training data from HotpotQA experiment outputs and append to a JSONL file.

Usage:
    python scripts/collect_training_data.py --results-dir results/hotpotqa --out data/train/hotpot_training_data.jsonl

The script will scan for files named `task_*.json` under `--results-dir` and
for each decomposer result produce one training record. By default it will
compute a question embedding using `sentence-transformers` if available and
include it in the output record under `question_embedding`.

Each call appends new records to the output file. Pass `--unique` to avoid
adding records whose `sample_id|decomposer` key already exists in the file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import glob

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def find_task_files(results_dir: Path) -> List[Path]:
    return sorted(Path(results_dir).rglob("task_*.json"))


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}")
        return None


def compute_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed; cannot compute embeddings")
    model = SentenceTransformer(model_name)
    # return native Python floats (not numpy.float32) so JSON serialization works
    arr = model.encode(texts, convert_to_numpy=True)
    try:
        return arr.tolist()
    except Exception:
        # fallback: ensure rows are converted element-wise to Python floats
        return [[float(v) for v in x] for x in arr]


def extract_records_from_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    sample_id = sample.get('sample_id')
    question = sample.get('question', '')
    decomposer_results = sample.get('decomposer_results', [])
    decomposer_scores = sample.get('decomposer_scores', {})
    per_details = sample.get('per_decomposer_details', [])
    shapley_map = sample.get('decomposer_shapley', {})
    baseline = sample.get('baseline_avgF1_per_token')
    total_tokens = sample.get('total_tokens')

    for dr in decomposer_results:
        did = dr.get('decomposer')
        sub_questions = dr.get('sub_questions', [])
        keywords = dr.get('keywords', []) or dr.get('pipeline_output', {}).get('keywords', []) or []
        final_answer = dr.get('final_answer', '')
        tokens_used = float(dr.get('tokens_used', 0.0) or 0.0)

        # try to find shapley_per_token in decomposer_scores or per_details
        shap_pt = None
        if did in decomposer_scores:
            shap_pt = decomposer_scores[did].get('components', {}).get('shapley_per_token')
        else:
            for pd in per_details:
                if pd.get('decomposer') == did:
                    shap_pt = pd.get('components', {}).get('shapley_per_token')
                    break

        shap = shapley_map.get(did)
        score = decomposer_scores.get(did, {}).get('score') if decomposer_scores else None

        rec = {
            'sample_id': sample_id,
            'question': question,
            'decomposer': did,
            'sub_questions': sub_questions,
            'keywords': keywords,
            'final_answer': final_answer,
            'tokens_used': tokens_used,
            'shapley': shap,
            'shapley_per_token': shap_pt,
            'score': score,
            'baseline_avgF1_per_token': baseline,
            'total_tokens': total_tokens,
        }
        records.append(rec)
    return records


def load_existing_keys(path: Path) -> set:
    keys = set()
    if not path.exists():
        return keys
    with open(path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                key = f"{data.get('sample_id')}|{data.get('decomposer')}"
                keys.add(key)
            except Exception:
                continue
    return keys


def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', type=str, default='results/hotpotqa', help='Directory with task_*.json files')
    p.add_argument('--out', type=str, default='data/train/hotpot_training_data.jsonl', help='Output JSONL file to append')
    p.add_argument('--embed', action='store_true', help='Compute question embeddings (requires sentence-transformers)')
    p.add_argument('--model', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model name')
    p.add_argument('--unique', action='store_true', help='Avoid appending duplicate sample_id|decomposer records')
    args = p.parse_args(argv)

    results_dir = Path(args.results_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = find_task_files(results_dir)
    if not files:
        print(f"No task_*.json files found under {results_dir}")
        return

    existing = set()
    if args.unique:
        existing = load_existing_keys(out_path)

    all_records: List[Dict[str, Any]] = []
    questions_to_embed: List[str] = []
    recs_for_embedding_indices: List[int] = []

    for fp in files:
        sample = load_json(fp)
        if not sample:
            continue
        recs = extract_records_from_sample(sample)
        for r in recs:
            key = f"{r.get('sample_id')}|{r.get('decomposer')}"
            if args.unique and key in existing:
                continue
            # remember for embedding
            all_records.append(r)
            questions_to_embed.append(r.get('question', ''))
            recs_for_embedding_indices.append(len(all_records)-1)

    # compute embeddings if requested
    embeddings = None
    if args.embed:
        if SentenceTransformer is None:
            print("sentence-transformers not available; skipping embeddings")
        else:
            print(f"Computing embeddings for {len(questions_to_embed)} questions...")
            emb_list = compute_embeddings(questions_to_embed, model_name=args.model)
            for idx, emb in enumerate(emb_list):
                rec_idx = recs_for_embedding_indices[idx]
                all_records[rec_idx]['question_embedding'] = emb

    # write appended records
    appended = 0
    with open(out_path, 'a') as out_f:
        for r in all_records:
            out_f.write(json.dumps(r) + "\n")
            appended += 1

    print(f"Appended {appended} records to {out_path}")


if __name__ == '__main__':
    main()
