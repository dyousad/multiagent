# Multi-Agent LLM Collaboration with Shapley Reward Allocation

A research framework for multi-agent collaboration using Large Language Models (LLMs) with Shapley value-based credit assignment.

## Overview

This project implements a multi-agent system where LLM-powered agents collaborate on tasks and receive credit allocation based on their marginal contributions, computed using Shapley values from cooperative game theory.

### Key Features

- **Multi-Agent Architecture**: Parallel and sequential execution modes with message passing
- **LLM Integration**: Support for Deepseek, Qwen, and other OpenAI-compatible APIs
- **Shapley Value Credit Assignment**: Fair reward allocation based on marginal contributions
- **Dynamic Credit Reallocation**: Test-time counterfactual-based credit updates
- **HotpotQA Multi-Hop QA**: Full support for multi-hop question answering evaluation
- **Question Decomposition**: Automatic breakdown of complex questions into sub-questions
- **Baseline Comparisons**: Uniform and proportional reward allocation methods
- **Comprehensive Evaluation**: Metrics including fairness index, reward variance, and accuracy
- **Visualization Tools**: Automated plotting of results and comparisons

### 🆕 New: HotpotQA Integration

This framework now includes complete support for evaluating multi-agent systems on **HotpotQA**, a challenging multi-hop question answering dataset. Features include:

- **Question Decomposer Agent**: Automatically breaks down complex questions into simpler sub-questions
- **Dynamic Credit Allocation**: Counterfactual-based credit assignment at test time
- **Comprehensive Metrics**: Exact Match, F1, Gini/Jain fairness indices, Shannon entropy
- **Rich Visualizations**: 5+ plot types comparing static vs dynamic credit allocation

**Quick Start for HotpotQA:**
```bash
# Run the quick start script
./run_hotpotqa.sh --samples 10 --agents 3

# Or run manually
python scripts/run_hotpotqa_experiments.py
python scripts/plot_hotpotqa_results.py
```

📖 See [HOTPOTQA_GUIDE.md](HOTPOTQA_GUIDE.md) for detailed documentation and [HOTPOTQA_SUMMARY.md](HOTPOTQA_SUMMARY.md) for implementation details.

## Project Structure

```
multiagent/
├── src/
│   ├── __init__.py
│   ├── agent.py                    # Base agent class
│   ├── llm_agent.py                # LLM-powered agent implementation
│   ├── environment.py              # Task environment
│   ├── environment_hotpotqa.py     # HotpotQA environment (NEW)
│   ├── decomposer_agent.py         # Question decomposer agent (NEW)
│   ├── evaluation.py               # Multi-hop QA metrics (NEW)
│   ├── controller.py               # Multi-agent controller with asyncio
│   ├── shapley.py                  # Shapley value computation
│   ├── reward_manager.py           # Reward allocation system (UPDATED)
│   └── main.py                     # Main entrypoint (UPDATED)
├── scripts/
│   ├── run_experiments.py          # Experiment runner
│   ├── run_hotpotqa_experiments.py # HotpotQA experiment runner (NEW)
│   ├── plot_results.py             # Visualization script
│   ├── plot_hotpotqa_results.py    # HotpotQA visualizations (NEW)
│   └── test_hotpotqa_integration.py # Integration tests (NEW)
├── data/                           # Dataset storage
├── models/                         # Model checkpoints/configs
├── results/                        # Experiment results
├── config.yaml                     # Configuration file (UPDATED)
├── run_hotpotqa.sh                 # Quick start script (NEW)
├── HOTPOTQA_GUIDE.md              # HotpotQA usage guide (NEW)
├── HOTPOTQA_SUMMARY.md            # Implementation summary (NEW)
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- Conda (recommended)

### Environment Setup

1. Clone the repository and navigate to the project directory:
```bash
cd /home/huatong/multiagent
```

2. Create a conda environment (or use existing `vlm-anchor`):
```bash
conda create -n multiagent python=3.9
conda activate multiagent
```

3. Install dependencies:
```bash
pip install requests matplotlib numpy
```

4. Set up API keys:
```bash
export DEEPSEEK_API_KEY="your-api-key-here"
# OR for other providers:
export OPENAI_API_KEY="your-openai-key"
```

## Usage

### Quick Start

Run a single experiment:

```bash
cd src
python main.py \
    --task "Write a Python function to calculate Fibonacci numbers" \
    --num_agents 3 \
    --model "deepseek-ai/DeepSeek-V3" \
    --reward shapley \
    --mode sequential
```

### Run Full Experiments

Compare all reward methods across multiple tasks:

```bash
python scripts/run_experiments.py
```

This will:
1. Run experiments with Shapley, uniform, and proportional reward methods
2. Save results to `results/experiments/`
3. Generate a comparison summary

### Visualize Results

Generate plots from experiment results:

```bash
python scripts/plot_results.py \
    --summary results/experiments/summary.json \
    --output_dir results/plots
```

## Configuration

### Model Selection

Supported model identifiers:

- **Deepseek**: `deepseek-ai/DeepSeek-V3`
- **Qwen**: `Qwen/Qwen2.5-7B-Instruct`
- **OpenAI**: `openai:gpt-3.5-turbo` or `openai:gpt-4`

### Execution Modes

- **parallel**: All agents act simultaneously
- **sequential**: Agents take turns, each seeing previous responses
- **message_passing**: Agents communicate via messages (experimental)

### Reward Methods

- **shapley**: Shapley value-based credit assignment (recommended)
- **uniform**: Equal reward for all agents (baseline)
- **proportional**: Reward proportional to individual contribution (baseline)

## Methodology

### Shapley Value Computation

Shapley values fairly distribute credit by considering all possible coalitions of agents. For agent $i$:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (n - |S| - 1)!}{n!} [v(S \cup \{i\}) - v(S)]$$

Where:
- $N$ is the set of all agents
- $S$ is a coalition (subset of agents)
- $v(S)$ is the value function (performance of coalition $S$)
- $n$ is the total number of agents

We use **Monte Carlo approximation** for scalability with large numbers of agents.

### Evaluation Metrics

- **Fairness Index** (Jain's fairness index): Measures equality of reward distribution
- **Reward Variance**: Variance in allocated rewards across agents
- **Task Accuracy**: Success rate of task completion
- **Mean Reward**: Average reward per agent

## Example Results

After running experiments, you can expect output like:

```
Task 1: Write a Python function to calculate factorial...
----------------------------------------------------------------------
Metric               shapley         uniform         proportional
----------------------------------------------------------------------
fairness_index       0.9234          1.0000          0.8512
variance             0.0045          0.0000          0.0123
mean_reward          0.3333          0.3333          0.3333
```

## Local Model Support

To use local models from `~/experiment/model`:

1. Create symbolic links:
```bash
ln -s ~/experiment/model/Llama-2-7b-hf models/llama-2-7b
ln -s ~/experiment/model/Qwen3-4B models/qwen3-4b
```

2. Modify `src/llm_agent.py` to support local model loading (HuggingFace Transformers)

## API Integration

The project uses the API wrapper from `/home/huatong/experiment/api_models.py`:

- **DeepseekSiliconflowModel**: Generic wrapper for Deepseek/Qwen via SiliconFlow
- **OpenAIModel**: OpenAI API wrapper
- **AnthropicModel**: Claude API wrapper

## Contributing

To extend the framework:

1. **Add new agent types**: Subclass `Agent` in `src/agent.py`
2. **Custom value functions**: Implement in `src/shapley.py`
3. **New evaluation metrics**: Add to `src/reward_manager.py`
4. **Additional tasks**: Edit `scripts/run_experiments.py`

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{multiagent_shapley,
  title={Multi-Agent LLM Collaboration with Shapley Reward Allocation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/multiagent}
}
```

## License

MIT License

## References

1. Shapley, L. S. (1953). "A value for n-person games." *Contributions to the Theory of Games*.
2. Castro, J., Gómez, D., & Tejada, J. (2009). "Polynomial calculation of the Shapley value based on sampling."
3. OpenAI et al. (2024). "Multi-agent collaboration with large language models."

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].
