#!/usr/bin/env python3
"""验证嵌入缓存的完整性和可用性"""

import sys
from pathlib import Path
import pickle
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def verify_embedding_cache():
    """验证嵌入缓存的完整性"""
    print("🔍 验证嵌入缓存完整性")
    print("="*60)

    cache_dir = Path("data/cache")
    model_name = "BAAI_bge-large-en-v1.5"

    # 预期的缓存文件
    expected_files = {
        'corpus': f"hotpotqa_corpus_full_{model_name}_corpus.pkl",
        'texts': f"hotpotqa_corpus_full_{model_name}_texts.pkl",
        'embeddings': f"hotpotqa_corpus_full_{model_name}_embeddings.npy",
        'index': f"hotpotqa_corpus_full_{model_name}_index.faiss"
    }

    print("📁 检查缓存文件:")
    existing_files = {}
    for key, filename in expected_files.items():
        filepath = cache_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / 1024 / 1024
            existing_files[key] = filepath
            print(f"   ✅ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {filename} (缺失)")

    if not existing_files:
        print("⚠️  未找到任何缓存文件！")
        return

    print(f"\n📊 缓存内容验证:")

    # 验证corpus缓存
    if 'corpus' in existing_files:
        try:
            with open(existing_files['corpus'], 'rb') as f:
                corpus = pickle.load(f)
            print(f"   Corpus: {len(corpus):,} 条记录")

            # 检查corpus结构
            if corpus and isinstance(corpus[0], dict):
                sample_doc = corpus[0]
                keys = list(sample_doc.keys())
                print(f"   Corpus结构: {keys}")
            else:
                print("   ⚠️  Corpus格式异常")
        except Exception as e:
            print(f"   ❌ Corpus读取失败: {e}")

    # 验证texts缓存
    if 'texts' in existing_files:
        try:
            with open(existing_files['texts'], 'rb') as f:
                texts = pickle.load(f)
            print(f"   Texts: {len(texts):,} 条记录")

            if texts:
                avg_length = sum(len(text) for text in texts[:1000]) / min(1000, len(texts))
                print(f"   平均文本长度: {avg_length:.0f} 字符")
        except Exception as e:
            print(f"   ❌ Texts读取失败: {e}")

    # 验证embeddings
    if 'embeddings' in existing_files:
        try:
            embeddings = np.load(existing_files['embeddings'])
            print(f"   Embeddings: {embeddings.shape}")
            print(f"   维度: {embeddings.shape[1]}")
            print(f"   数据类型: {embeddings.dtype}")
        except Exception as e:
            print(f"   ❌ Embeddings读取失败: {e}")

    # 验证FAISS索引
    if 'index' in existing_files:
        try:
            import faiss
            index = faiss.read_index(str(existing_files['index']))
            print(f"   FAISS��引: {index.ntotal} 向量")
            print(f"   索引维度: {index.d}")
            print(f"   索引类型: {type(index).__name__}")
        except ImportError:
            print(f"   ⚠️  FAISS未安装，无法验证索引")
        except Exception as e:
            print(f"   ❌ FAISS索引读取失败: {e}")

    # 一致性检查
    print(f"\n🔍 一致性检查:")
    counts = {}

    if 'corpus' in existing_files:
        counts['corpus'] = len(corpus)
    if 'texts' in existing_files:
        counts['texts'] = len(texts)
    if 'embeddings' in existing_files:
        counts['embeddings'] = embeddings.shape[0]
    if 'index' in existing_files and 'index' in locals():
        counts['index'] = index.ntotal

    if len(set(counts.values())) == 1:
        print(f"   ✅ 所有缓存文件数量一致: {list(counts.values())[0]:,}")
    else:
        print(f"   ⚠️  缓存文件数量不一致:")
        for key, count in counts.items():
            print(f"      {key}: {count:,}")

    # 测试检索功能
    print(f"\n🧪 功能测试:")
    try:
        from cached_retrieval_manager import CachedRetrievalManager

        print("   初始化CachedRetrievalManager...")
        manager = CachedRetrievalManager(
            corpus_path="data/hotpotqa_corpus_full.json",
            model_name="BAAI/bge-large-en-v1.5",
            cache_dir="data/cache"
        )

        print(f"   Manager加载成功: {len(manager.corpus):,} 文档")

        # 测试检索
        test_query = "Scott Derrickson nationality"
        results = manager.retrieve(test_query, top_k=3)

        print(f"   测试检索: '{test_query}'")
        print(f"   返回结果: {len(results)} 条")
        if results:
            print(f"   首个结果: {results[0][:100]}...")
            print("   ✅ 检索功能正常")
        else:
            print("   ⚠️  检索无结果")

    except Exception as e:
        print(f"   ❌ 功能测试失败: {e}")

    print(f"\n" + "="*60)

if __name__ == "__main__":
    verify_embedding_cache()