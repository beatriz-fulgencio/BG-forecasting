#!/bin/bash

# Blood Glucose Forecasting Benchmark CLI Runner
# This script provides common CLI commands for running the benchmark framework

set -e  # Exit on any error

# Color codes for output formatting
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

# Function to run CLI command with error handling
run_cli() {
    local description="$1"
    shift
    print_status "Running: $description"
    echo "Command: python benchmark/cli.py $@"
    echo "----------------------------------------"
    
    if python benchmark/cli.py "$@"; then
        print_success "Completed: $description"
    else
        print_error "Failed: $description"
        exit 1
    fi
    echo
}

# Display usage information
show_usage() {
    echo "Blood Glucose Forecasting Benchmark CLI Runner"
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Available commands:"
    echo "  quick          - Quick test with GRU on patient 540 (3 epochs)"
    echo "  demo           - Demo run with multiple models and patients"
    echo "  full           - Full benchmark with all models and patients"
    echo "  transfer       - Transfer learning example"
    echo "  no-transfer    - Training without transfer learning"
    echo "  custom         - Custom configuration example"
    echo "  compare        - Compare experiment results"
    echo "  data-list      - List available datasets"
    echo "  data-process   - Process raw data"
    echo "  help           - Show this help message"
    echo
    echo "Examples:"
    echo "  $0 quick        # Quick test"
    echo "  $0 demo         # Demo with multiple models"
    echo "  $0 custom       # Custom configuration"
}

# Command implementations
run_quick() {
    print_status "Running quick test..."
    run_cli "Quick GRU test on patient 540" \
        run --patients 540 --models gru --epochs 3 --no-transfer-learning --name "quick_test"
}

run_demo() {
    print_status "Running demo with multiple models..."
    run_cli "Demo with RNN, LSTM, GRU on patients 540, 544" \
        run --patients 540 544 --models rnn lstm gru --epochs 5 --name "demo_run"
}

run_full() {
    print_status "Running full benchmark..."
    run_cli "Full benchmark with all models and patients" \
        run --models rnn lstm gru --epochs 10 --name "full_benchmark"
}

run_transfer() {
    print_status "Running transfer learning example..."
    run_cli "Transfer learning with GRU" \
        run --patients 540 544 --models gru --epochs 10 --name "transfer_learning_example"
}

run_no_transfer() {
    print_status "Running without transfer learning..."
    run_cli "Training without transfer learning" \
        run --patients 540 --models gru --epochs 8 --no-transfer-learning --name "no_transfer_example"
}

run_custom() {
    print_status "Running custom configuration..."
    run_cli "Custom configuration with specific parameters" \
        run --patients 540 544 \
             --models lstm gru \
             --epochs 7 \
             --batch-size 32 \
             --learning-rate 0.001 \
             --sequence-length 12 \
             --prediction-horizon 6 \
             --device auto \
             --name "custom_config" \
             --tags experiment custom high-lr
}

run_compare() {
    print_status "Comparing experiment results..."
    
    # Find the most recent result directories
    result_dirs=($(ls -dt results/experiment_* 2>/dev/null | head -2))
    
    if [ ${#result_dirs[@]} -lt 2 ]; then
        print_warning "Need at least 2 experiment results to compare"
        print_status "Available results:"
        ls -la results/ 2>/dev/null || echo "No results directory found"
        return 1
    fi
    
    run_cli "Comparing latest experiments" \
        compare "${result_dirs[@]}" --output "results/comparison_$(date +%Y%m%d_%H%M%S).json"
}

run_data_list() {
    print_status "Listing available datasets..."
    run_cli "List datasets" data list
}

run_data_process() {
    print_status "Processing raw data..."
    run_cli "Process data for patients 540, 544" \
        data process --patients 540 544
}

# Interactive mode
run_interactive() {
    echo "Blood Glucose Forecasting Benchmark - Interactive Mode"
    echo "======================================================"
    echo
    echo "Select an option:"
    echo "1) Quick test (GRU, patient 540, 3 epochs)"
    echo "2) Demo run (multiple models, patients 540,544)"
    echo "3) Full benchmark (all models, 10 epochs)"
    echo "4) Transfer learning example"
    echo "5) No transfer learning example"
    echo "6) Custom configuration"
    echo "7) Compare results"
    echo "8) List datasets"
    echo "9) Process raw data"
    echo "0) Exit"
    echo
    read -p "Enter your choice (0-9): " choice
    
    case $choice in
        1) run_quick ;;
        2) run_demo ;;
        3) run_full ;;
        4) run_transfer ;;
        5) run_no_transfer ;;
        6) run_custom ;;
        7) run_compare ;;
        8) run_data_list ;;
        9) run_data_process ;;
        0) echo "Goodbye!"; exit 0 ;;
        *) print_error "Invalid choice. Please try again."; run_interactive ;;
    esac
}

# Main script logic
main() {
    # Check if we're in the right directory
    if [ ! -f "benchmark/cli.py" ]; then
        print_error "CLI script not found. Please run this script from the project root directory."
        exit 1
    fi
    
    # Parse command line arguments
    case "${1:-}" in
        "quick"|"q")
            run_quick
            ;;
        "demo"|"d")
            run_demo
            ;;
        "full"|"f")
            run_full
            ;;
        "transfer"|"t")
            run_transfer
            ;;
        "no-transfer"|"nt")
            run_no_transfer
            ;;
        "custom"|"c")
            run_custom
            ;;
        "compare"|"comp")
            run_compare
            ;;
        "data-list"|"dl")
            run_data_list
            ;;
        "data-process"|"dp")
            run_data_process
            ;;
        "help"|"h"|"--help"|"-h")
            show_usage
            ;;
        "")
            run_interactive
            ;;
        *)
            print_error "Unknown command: $1"
            echo
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
