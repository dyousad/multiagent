#!/usr/bin/env python3
"""Monitor the progress of the full enhanced experiment."""

import time
import json
from pathlib import Path

def monitor_experiment_progress():
    """Monitor the experiment progress."""
    output_dir = Path("results/enhanced_evidence_full")

    print("📊 Monitoring Enhanced Evidence Collection Experiment")
    print("=" * 60)

    start_time = time.time()
    last_sample_count = 0

    while True:
        try:
            # Check if experiment is complete
            results_file = output_dir / "hotpotqa_results.json"

            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)

                samples = results.get("samples", [])
                current_sample_count = len(samples)

                if current_sample_count != last_sample_count:
                    elapsed = time.time() - start_time
                    remaining_samples = 100 - current_sample_count

                    if current_sample_count > 0:
                        avg_time_per_sample = elapsed / current_sample_count
                        estimated_remaining = remaining_samples * avg_time_per_sample

                        print(f"\n⏱️ Progress Update at {time.strftime('%H:%M:%S')}")
                        print(f"  Samples completed: {current_sample_count}/100 ({current_sample_count}%)")
                        print(f"  Elapsed time: {elapsed/60:.1f} minutes")
                        print(f"  Avg time per sample: {avg_time_per_sample:.1f}s")
                        print(f"  Estimated remaining: {estimated_remaining/60:.1f} minutes")
                        print(f"  Estimated completion: {time.strftime('%H:%M:%S', time.localtime(time.time() + estimated_remaining))}")

                    last_sample_count = current_sample_count

                if current_sample_count >= 100:
                    print(f"\n✅ Experiment completed!")

                    # Show quick summary
                    aggregate = results.get("aggregate", {})
                    if aggregate:
                        print(f"📊 Quick Results:")
                        print(f"  Exact Match: {aggregate.get('exact_match_accuracy', 0)*100:.1f}%")
                        print(f"  Average F1: {aggregate.get('average_f1', 0):.3f}")

                    break

            # Check individual task files for more detailed progress
            task_files = list(output_dir.glob("task_*.json"))
            if task_files:
                completed_tasks = len(task_files)
                if completed_tasks != last_sample_count:
                    print(f"📁 Task files created: {completed_tasks}")

                    # Quick analysis of recent completions
                    if completed_tasks >= 5:
                        recent_files = sorted(task_files, key=lambda x: x.stat().st_mtime)[-5:]
                        unknown_count = 0
                        correct_count = 0

                        for file in recent_files:
                            try:
                                with open(file, 'r') as f:
                                    task_data = json.load(f)
                                    if task_data.get("predicted_answer", "").lower() == "unknown":
                                        unknown_count += 1
                                    if task_data.get("exact_match", False):
                                        correct_count += 1
                            except:
                                continue

                        print(f"  Recent 5 samples: {correct_count}/5 correct, {unknown_count}/5 unknown")

            if not results_file.exists() and not task_files:
                elapsed = time.time() - start_time
                print(f"\n🔄 Experiment starting... (elapsed: {elapsed:.0f}s)")
                if elapsed > 300:  # 5 minutes
                    print(f"  Note: Initial setup may take several minutes for model loading and retrieval system initialization")

            time.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            print(f"\n⏹️ Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n⚠️ Monitoring error: {e}")
            time.sleep(60)  # Wait longer on error

if __name__ == "__main__":
    monitor_experiment_progress()