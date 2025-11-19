#!/usr/bin/env python3
"""
Unified RAG Enhancement Test Script

This script consolidates all RAG testing functionality into a single interface,
defaulting to complex reasoning (query expansion + multi-hop retrieval + quality filtering)
while providing options for different test modes and configurations.

Features:
- Complex reasoning as default (full enhancement pipeline)
- Multiple test modes: quick, standard, comprehensive
- Automatic environment detection and setup
- Simplified command-line interface
- Consolidated results reporting
"""

import json
import os
import sys
import time
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src'))

from rag_enhanced_agent import RAGEnhancedAgent
from multi_hop_retriever import MultiHopConfig
from evidence_quality_filter import FilterConfig

class UnifiedRAGTester:
    """Unified RAG enhancement testing system"""

    def __init__(self, model_identifier: str = "deepseek-ai/DeepSeek-V3"):
        self.model_identifier = model_identifier
        self.results_dir = Path("results/unified_rag_test")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def create_complex_reasoning_agent(self, agent_id: str = "complex_reasoner") -> RAGEnhancedAgent:
        """Create agent with complex reasoning (full enhancement) as default"""

        # Optimized configuration based on comprehensive testing results
        multi_hop_config = MultiHopConfig(
            max_hops=3,
            evidence_per_hop=4,
            min_confidence_threshold=0.25,
            entity_overlap_threshold=0.15,
            enable_backward_validation=True,
            max_follow_up_queries=3
        )

        quality_filter_config = FilterConfig(
            min_content_score=0.25,
            min_relevance_score=0.3,
            max_contradiction_penalty=0.7,
            enable_entity_consistency=True,
            enable_temporal_filtering=True,
            enable_contradiction_detection=True,
            diversity_threshold=0.8
        )

        agent = RAGEnhancedAgent(
            agent_id=agent_id,
            model_identifier=self.model_identifier,
            role="complex reasoning researcher",
            top_k=5,
            use_query_expansion=True,      # ✓ Query expansion
            use_multi_hop=True,            # ✓ Multi-hop retrieval
            use_quality_filter=True,       # ✓ Quality filtering
            multi_hop_config=multi_hop_config,
            quality_filter_config=quality_filter_config
        )

        print(f"🧠 Created complex reasoning agent: {agent.agent_id}")
        print("   ✓ Query expansion enabled")
        print("   ✓ Multi-hop retrieval enabled (max 3 hops)")
        print("   ✓ Evidence quality filtering enabled")

        return agent

    def create_baseline_agent(self, agent_id: str = "baseline") -> RAGEnhancedAgent:
        """Create baseline agent for comparison (minimal enhancements)"""
        agent = RAGEnhancedAgent(
            agent_id=agent_id,
            model_identifier=self.model_identifier,
            role="baseline researcher",
            top_k=5,
            use_query_expansion=False,
            use_multi_hop=False,
            use_quality_filter=False
        )

        print(f"📊 Created baseline agent: {agent.agent_id}")
        return agent

    def get_test_questions(self, mode: str = "standard") -> List[Dict[str, Any]]:
        """Get test questions based on mode"""

        # Core question set for all modes
        questions = [
            {
                "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
                "type": "comparison",
                "complexity": "medium",
                "expected_hops": ["Scott Derrickson nationality", "Ed Wood nationality"]
            },
            {
                "question": "What government position was held by the woman who portrayed Corliss Archer in Kiss and Tell?",
                "type": "multi_hop_entity",
                "complexity": "high",
                "expected_hops": ["actress who played Corliss Archer", "government position"]
            },
            {
                "question": "What is the seating capacity of the arena where the Lewiston Maineiacs played?",
                "type": "multi_hop_factual",
                "complexity": "high",
                "expected_hops": ["Lewiston Maineiacs arena", "seating capacity"]
            }
        ]

        if mode in ["standard", "comprehensive"]:
            questions.extend([
                {
                    "question": "The director of Big Stone Gap is based in what New York city?",
                    "type": "entity_location",
                    "complexity": "medium",
                    "expected_hops": ["Big Stone Gap director", "director location"]
                },
                {
                    "question": "What science fantasy young adult series has companion books about enslaved worlds?",
                    "type": "descriptive_search",
                    "complexity": "medium",
                    "expected_hops": ["science fantasy series", "companion books"]
                }
            ])

        if mode == "comprehensive":
            questions.extend([
                {
                    "question": "What is the seating capacity of the arena where the team that won the 1995 Stanley Cup Finals played their home games?",
                    "type": "complex_multi_hop",
                    "complexity": "very_high",
                    "expected_hops": ["1995 Stanley Cup winner", "home arena", "seating capacity"]
                },
                {
                    "question": "In what year was the university founded that is located in the same city as the company that developed the iPhone?",
                    "type": "complex_multi_hop",
                    "complexity": "very_high",
                    "expected_hops": ["iPhone developer", "company city", "university in city", "founding year"]
                }
            ])

        return questions

    def run_test_mode(self, mode: str = "standard", compare_baseline: bool = True) -> Dict[str, Any]:
        """Run tests in specified mode"""

        print(f"🔬 Running {mode.upper()} mode test")
        print("=" * 60)

        questions = self.get_test_questions(mode)
        results = {
            "mode": mode,
            "complex_reasoning_results": [],
            "baseline_results": [] if compare_baseline else None,
            "comparison": {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Create agents
        complex_agent = self.create_complex_reasoning_agent()
        baseline_agent = self.create_baseline_agent() if compare_baseline else None

        print(f"\n📋 Testing {len(questions)} questions...")
        print()

        # Test each question
        for i, question_data in enumerate(questions, 1):
            question = question_data["question"]
            complexity = question_data["complexity"]

            print(f"Question {i}/{len(questions)} ({complexity.upper()})")
            print(f"Q: {question}")
            print()

            # Test complex reasoning agent
            print("🧠 Complex Reasoning Agent:")
            start_time = time.time()
            complex_result = complex_agent.retrieve_evidence(question)
            complex_time = time.time() - start_time

            evidence_count = complex_result.get('num_retrieved', 0)
            hop_count = complex_result.get('hop_count', 1)
            quality_used = complex_result.get('quality_filter_used', False)
            quality_score = complex_result.get('avg_quality_score', 0)

            print(f"  📊 Evidence retrieved: {evidence_count}")
            print(f"  🔄 Hops performed: {hop_count}")
            if quality_used:
                print(f"  🎯 Avg quality score: {quality_score:.3f}")
            print(f"  ⏱️  Time: {complex_time:.2f}s")
            print()

            # Store complex reasoning results
            results["complex_reasoning_results"].append({
                "question": question,
                "complexity": complexity,
                "evidence_count": evidence_count,
                "hop_count": hop_count,
                "quality_score": quality_score,
                "retrieval_time": complex_time,
                "evidence_preview": complex_result.get('evidence', [])[:2]
            })

            # Test baseline agent if enabled
            if compare_baseline and baseline_agent:
                print("📊 Baseline Agent:")
                start_time = time.time()
                baseline_result = baseline_agent.retrieve_evidence(question)
                baseline_time = time.time() - start_time

                baseline_count = baseline_result.get('num_retrieved', 0)
                print(f"  📊 Evidence retrieved: {baseline_count}")
                print(f"  ⏱️  Time: {baseline_time:.2f}s")

                # Store baseline results
                results["baseline_results"].append({
                    "question": question,
                    "evidence_count": baseline_count,
                    "retrieval_time": baseline_time,
                    "evidence_preview": baseline_result.get('evidence', [])[:2]
                })

                # Quick comparison
                improvement = evidence_count - baseline_count
                print(f"  📈 Evidence improvement: {improvement:+d}")

            print("-" * 60)
            print()

        # Calculate summary statistics
        complex_results = results["complex_reasoning_results"]
        avg_evidence = sum(r['evidence_count'] for r in complex_results) / len(complex_results)
        avg_time = sum(r['retrieval_time'] for r in complex_results) / len(complex_results)
        avg_hops = sum(r['hop_count'] for r in complex_results) / len(complex_results)
        avg_quality = sum(r['quality_score'] for r in complex_results) / len(complex_results)

        results["comparison"] = {
            "complex_reasoning": {
                "avg_evidence": avg_evidence,
                "avg_time": avg_time,
                "avg_hops": avg_hops,
                "avg_quality": avg_quality
            }
        }

        if compare_baseline and results["baseline_results"]:
            baseline_results = results["baseline_results"]
            baseline_avg_evidence = sum(r['evidence_count'] for r in baseline_results) / len(baseline_results)
            baseline_avg_time = sum(r['retrieval_time'] for r in baseline_results) / len(baseline_results)

            results["comparison"]["baseline"] = {
                "avg_evidence": baseline_avg_evidence,
                "avg_time": baseline_avg_time
            }

            results["comparison"]["improvement"] = {
                "evidence_gain": avg_evidence - baseline_avg_evidence,
                "time_overhead": avg_time - baseline_avg_time,
                "evidence_improvement_pct": ((avg_evidence - baseline_avg_evidence) / baseline_avg_evidence * 100) if baseline_avg_evidence > 0 else 0
            }

        # Print summary
        self.print_test_summary(results)

        return results

    def print_test_summary(self, results: Dict[str, Any]):
        """Print comprehensive test summary"""

        print("📊 TEST SUMMARY")
        print("=" * 60)

        complex_stats = results["comparison"]["complex_reasoning"]
        print(f"🧠 Complex Reasoning Performance:")
        print(f"   • Average evidence: {complex_stats['avg_evidence']:.1f}")
        print(f"   • Average hops: {complex_stats['avg_hops']:.1f}")
        print(f"   • Average quality: {complex_stats['avg_quality']:.3f}")
        print(f"   • Average time: {complex_stats['avg_time']:.2f}s")

        if "improvement" in results["comparison"]:
            improvement = results["comparison"]["improvement"]
            print(f"\n📈 vs Baseline Improvement:")
            print(f"   • Evidence gain: {improvement['evidence_gain']:+.1f}")
            print(f"   • Improvement: {improvement['evidence_improvement_pct']:+.1f}%")
            print(f"   • Time overhead: {improvement['time_overhead']:+.2f}s")

        print(f"\n💡 Recommendation:")
        if complex_stats['avg_evidence'] >= 4 and complex_stats['avg_quality'] >= 0.5:
            print("   ✅ Complex reasoning shows excellent performance")
            print("   ✅ Recommended for production use")
        elif complex_stats['avg_evidence'] >= 3:
            print("   ✅ Complex reasoning shows good performance")
            print("   ⚠️  Monitor quality scores for optimization")
        else:
            print("   ⚠️  Complex reasoning needs tuning")
            print("   💡 Consider adjusting thresholds")

    def save_results(self, results: Dict[str, Any], filename_suffix: str = ""):
        """Save results to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"unified_test_results_{timestamp}{filename_suffix}.json"
        filepath = self.results_dir / filename

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"📁 Results saved to: {filepath}")
        return filepath

def main():
    """Main entry point with argument parsing"""

    parser = argparse.ArgumentParser(
        description="Unified RAG Enhancement Test - Complex Reasoning by Default",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unified_rag_test.py                    # Standard test with complex reasoning
  python unified_rag_test.py --mode quick       # Quick test (3 questions)
  python unified_rag_test.py --mode comprehensive   # Full comprehensive test
  python unified_rag_test.py --no-baseline      # Skip baseline comparison
  python unified_rag_test.py --model gpt-4      # Use different model
        """
    )

    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "comprehensive"],
        default="standard",
        help="Test mode: quick (3 questions), standard (5 questions), comprehensive (7 questions)"
    )

    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-V3",
        help="Model identifier to use for testing"
    )

    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline comparison (faster execution)"
    )

    parser.add_argument(
        "--save-results",
        action="store_true",
        default=True,
        help="Save detailed results to JSON file"
    )

    args = parser.parse_args()

    print("🧪 Unified RAG Enhancement Test")
    print("🧠 Complex Reasoning by Default (Query Expansion + Multi-hop + Quality Filter)")
    print("=" * 70)
    print(f"Mode: {args.mode.upper()}")
    print(f"Model: {args.model}")
    print(f"Baseline comparison: {'No' if args.no_baseline else 'Yes'}")
    print()

    try:
        # Initialize tester
        tester = UnifiedRAGTester(model_identifier=args.model)

        # Run tests
        results = tester.run_test_mode(
            mode=args.mode,
            compare_baseline=not args.no_baseline
        )

        # Save results if requested
        if args.save_results:
            suffix = f"_{args.mode}"
            if args.no_baseline:
                suffix += "_no_baseline"
            tester.save_results(results, suffix)

        print("\n🎉 Unified RAG test completed successfully!")

        return results

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()