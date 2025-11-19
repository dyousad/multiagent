#!/usr/bin/env python3
"""全面验证corpus和缓存的完整性"""

import json
import pickle
from pathlib import Path

def analyze_corpus_completeness():
    """分析corpus和缓存的完整性"""
    print("="*80)
    print("HotpotQA Corpus & Cache 完整性分析")
    print("="*80)
    print()

    # 1. 分析���始数据
    print("📊 原始数据集分析:")
    with open('data/hotpot_dev_fullwiki_v1.json') as f:
        hotpotqa_data = json.load(f)

    total_samples = len(hotpotqa_data)
    total_contexts = 0
    total_sentences = 0
    unique_titles = set()

    for sample in hotpotqa_data:
        context = sample.get('context', [])
        total_contexts += len(context)

        for title, sentences in context:
            unique_titles.add(title)
            if isinstance(sentences, list):
                total_sentences += len(sentences)
            else:
                total_sentences += 1

    print(f"   - HotpotQA样本数: {total_samples:,}")
    print(f"   - Context对数: {total_contexts:,}")
    print(f"   - 唯一标题数: {len(unique_titles):,}")
    print(f"   - 预计句子数: {total_sentences:,}")

    # 2. 分析corpus文件
    print(f"\n📁 Corpus文件分析:")
    corpus_files = {
        'hotpotqa_corpus.json': 'data/hotpotqa_corpus.json',
        'hotpotqa_corpus_large.json': 'data/hotpotqa_corpus_large.json',
        'hotpotqa_corpus_full.json': 'data/hotpotqa_corpus_full.json'
    }

    for name, path in corpus_files.items():
        if Path(path).exists():
            with open(path) as f:
                data = json.load(f)
            size_mb = Path(path).stat().st_size / 1024 / 1024
            coverage = len(data) / total_sentences * 100
            print(f"   - {name}:")
            print(f"     记录数: {len(data):,} ({coverage:.1f}% 覆盖率)")
            print(f"     文件大小: {size_mb:.1f} MB")

    # 3. 分析缓存
    print(f"\n💾 缓存文件分析:")
    cache_dir = Path("data/cache")
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.pkl"))
        print(f"   - 缓存文件数: {len(cache_files)}")

        for cache_file in cache_files:
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                size_mb = cache_file.stat().st_size / 1024 / 1024
                print(f"   - {cache_file.name}:")
                print(f"     记录数: {len(data):,}")
                print(f"     文件大小: {size_mb:.1f} MB")
            except Exception as e:
                print(f"   - {cache_file.name}: 读取失败 ({e})")

    # 4. 完整性检查
    print(f"\n🔍 完整性检查:")

    # 检查是否基于完整数据集
    full_corpus_path = "data/hotpotqa_corpus_full.json"
    if Path(full_corpus_path).exists():
        with open(full_corpus_path) as f:
            full_corpus = json.load(f)

        print(f"   - 完整corpus记录数: {len(full_corpus):,}")
        print(f"   - 覆盖率: {len(full_corpus)/total_sentences*100:.1f}%")

        if len(full_corpus) >= total_sentences * 0.85:  # 考虑去重和过滤
            print(f"   ✅ 覆盖率良好 (≥85%)")
        else:
            print(f"   ⚠️  覆盖率可能不足")

        # 检查缓存是否基于完整corpus
        corpus_cache = "data/cache/hotpotqa_corpus_full_BAAI_bge-large-en-v1.5_corpus.pkl"
        texts_cache = "data/cache/hotpotqa_corpus_full_BAAI_bge-large-en-v1.5_texts.pkl"

        if Path(corpus_cache).exists() and Path(texts_cache).exists():
            with open(corpus_cache, 'rb') as f:
                cached_corpus = pickle.load(f)
            with open(texts_cache, 'rb') as f:
                cached_texts = pickle.load(f)

            print(f"   - 缓存corpus记录数: {len(cached_corpus):,}")
            print(f"   - 缓存texts记录数: {len(cached_texts):,}")

            if len(cached_corpus) == len(full_corpus):
                print(f"   ✅ 缓存与完整corpus一致")
            else:
                print(f"   ⚠️  缓存与corpus大小不匹配")
                print(f"      差异: {abs(len(cached_corpus) - len(full_corpus)):,}条记录")

    # 5. 潜在问题检查
    print(f"\n🚨 潜在问题检查:")

    issues = []

    # 检查是否缺少其他数据源
    if total_sentences - len(full_corpus) > total_sentences * 0.15:
        issues.append("可能缺少大量句子数据")

    # 检查缓存模型
    bge_cache_exists = any("bge-large-en-v1.5" in str(f) for f in cache_dir.glob("*.pkl"))
    if not bge_cache_exists:
        issues.append("缺少BGE embedding模型的缓存")

    # 检查是否有多个corpus版本
    small_corpus_exists = Path("data/hotpotqa_corpus.json").exists()
    if small_corpus_exists:
        issues.append("存在多个corpus版本，可能导致混淆")

    if not issues:
        print(f"   ✅ 未发现明显问题")
    else:
        for issue in issues:
            print(f"   ⚠️  {issue}")

    # 6. 建议
    print(f"\n💡 建议:")
    print(f"   1. 使用 'hotpotqa_corpus_full.json' 作为主要corpus")
    print(f"   2. 确保RAG系统指向完整缓存文件")
    print(f"   3. 考虑删除较小的corpus文件避免混淆")
    print(f"   4. 缓存已基于完整数据集生成，可直接使用")

    print(f"\n" + "="*80)
    print(f"✅ 分析完成")
    print(f"="*80)

if __name__ == "__main__":
    analyze_corpus_completeness()