#!/bin/bash
# Quick launcher for multi-agent experiments

set -e

echo "=================================================="
echo "Multi-Agent LLM Collaboration System"
echo "=================================================="
echo ""

# Check if API key is set
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "Warning: DEEPSEEK_API_KEY not set!"
    echo "Please set your API key:"
    echo "  export DEEPSEEK_API_KEY='your-key-here'"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Default values
TASK="Write a Python function to calculate Fibonacci numbers"
NUM_AGENTS=3
MODEL="deepseek-ai/DeepSeek-V3"
REWARD="shapley"
MODE="sequential"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --agents)
            NUM_AGENTS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --reward)
            REWARD="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --task TEXT       Task description"
            echo "  --agents N        Number of agents (default: 3)"
            echo "  --model TEXT      Model identifier (default: deepseek-ai/DeepSeek-V3)"
            echo "  --reward METHOD   Reward method: shapley, uniform, proportional (default: shapley)"
            echo "  --mode MODE       Execution mode: parallel, sequential (default: sequential)"
            echo "  --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run.sh --task \"Implement quicksort\" --agents 4"
            echo "  ./run.sh --model \"Qwen/Qwen2.5-7B-Instruct\" --reward uniform"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Task: $TASK"
echo "  Agents: $NUM_AGENTS"
echo "  Model: $MODEL"
echo "  Reward: $REWARD"
echo "  Mode: $MODE"
echo ""

cd src
python main.py \
    --task "$TASK" \
    --num_agents "$NUM_AGENTS" \
    --model "$MODEL" \
    --reward "$REWARD" \
    --mode "$MODE"
