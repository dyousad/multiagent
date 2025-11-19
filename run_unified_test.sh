#!/bin/bash
# Unified RAG Enhancement Test - Shell Wrapper
#
# This script provides a simplified interface for running RAG enhancement tests
# with complex reasoning (query expansion + multi-hop + quality filtering) as default.
#
# Usage examples:
#   ./run_unified_test.sh                     # Standard test
#   ./run_unified_test.sh quick               # Quick test
#   ./run_unified_test.sh comprehensive       # Full comprehensive test
#   ./run_unified_test.sh standard --no-baseline  # Standard without baseline

set -e  # Exit on any error

# Default values
MODE="standard"
CONDA_ENV="vlm-anchor"
PYTHON_SCRIPT="unified_rag_test.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
🧪 Unified RAG Enhancement Test - Complex Reasoning by Default

Usage: $0 [mode] [options]

MODES:
    quick          Run quick test (3 questions, ~2 min)
    standard       Run standard test (5 questions, ~4 min) [DEFAULT]
    comprehensive  Run comprehensive test (7 questions, ~6 min)

OPTIONS:
    --no-baseline     Skip baseline comparison (faster)
    --model MODEL     Use specific model (default: deepseek-ai/DeepSeek-V3)
    --help, -h        Show this help message

FEATURES:
    ✓ Complex reasoning by default (Query Expansion + Multi-hop + Quality Filter)
    ✓ Automatic environment detection (vlm-anchor conda environment)
    ✓ Consolidated results reporting
    ✓ Performance comparison with baseline

EXAMPLES:
    $0                                    # Standard test with complex reasoning
    $0 quick                              # Quick test (3 questions)
    $0 comprehensive                      # Full test suite
    $0 standard --no-baseline             # Standard test without baseline
    $0 quick --model "gpt-4"              # Quick test with GPT-4

Environment: Uses conda environment '$CONDA_ENV'
EOF
}

# Parse command line arguments
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        quick|standard|comprehensive)
            MODE="$1"
            shift
            ;;
        --no-baseline)
            EXTRA_ARGS="$EXTRA_ARGS --no-baseline"
            shift
            ;;
        --model)
            if [[ -z "$2" ]]; then
                print_error "Model name required after --model"
                exit 1
            fi
            EXTRA_ARGS="$EXTRA_ARGS --model \"$2\""
            shift 2
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_warning "Unknown argument: $1"
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Header
echo "🧪 Unified RAG Enhancement Test"
echo "🧠 Complex Reasoning by Default"
echo "================================"
echo ""

# Check if conda environment exists
print_status "Checking conda environment: $CONDA_ENV"
if ! conda env list | grep -q "^$CONDA_ENV "; then
    print_error "Conda environment '$CONDA_ENV' not found!"
    print_status "Available environments:"
    conda env list
    echo ""
    print_status "To create the environment, run:"
    echo "  conda create -n $CONDA_ENV python=3.9"
    echo "  conda activate $CONDA_ENV"
    echo "  pip install -r requirements.txt"
    exit 1
fi

print_success "Conda environment '$CONDA_ENV' found"

# Check if Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    print_error "Python script '$PYTHON_SCRIPT' not found!"
    print_status "Make sure you're running this from the correct directory"
    exit 1
fi

print_success "Python script '$PYTHON_SCRIPT' found"

# Construct full command
FULL_COMMAND="conda run -n $CONDA_ENV python3 $PYTHON_SCRIPT --mode $MODE $EXTRA_ARGS"

print_status "Running mode: $MODE"
if [[ -n "$EXTRA_ARGS" ]]; then
    print_status "Extra arguments: $EXTRA_ARGS"
fi

print_status "Executing: $FULL_COMMAND"
echo ""

# Execute the command
print_status "Starting test execution..."
echo ""

if eval $FULL_COMMAND; then
    echo ""
    print_success "Test completed successfully!"

    # Show quick summary of what was done
    echo ""
    echo "📋 Test Summary:"
    echo "   • Mode: $MODE"
    echo "   • Environment: $CONDA_ENV"
    echo "   • Complex reasoning: ✓ Enabled (Query Expansion + Multi-hop + Quality Filter)"

    # Check if results directory exists and show latest results
    if [[ -d "results/unified_rag_test" ]]; then
        LATEST_RESULT=$(ls -t results/unified_rag_test/unified_test_results_*.json 2>/dev/null | head -1)
        if [[ -n "$LATEST_RESULT" ]]; then
            echo "   • Results saved: $LATEST_RESULT"
        fi
    fi

    echo ""
    echo "🎉 Unified RAG enhancement test completed!"
    echo ""
    echo "💡 Next steps:"
    echo "   • Review results in results/unified_rag_test/"
    echo "   • Try different modes: quick, standard, comprehensive"
    echo "   • Compare with --no-baseline for faster runs"

else
    echo ""
    print_error "Test failed!"

    echo ""
    echo "🔧 Troubleshooting:"
    echo "   1. Check if all dependencies are installed in '$CONDA_ENV'"
    echo "   2. Verify that the retrieval data is properly set up"
    echo "   3. Check conda environment: conda activate $CONDA_ENV"
    echo "   4. Run with --help for usage information"

    exit 1
fi