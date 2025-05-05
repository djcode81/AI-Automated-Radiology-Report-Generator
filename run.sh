#!/bin/bash
# CT Scan Radiology Report Generator - Complete Runner Script
# This script provides an all-in-one solution for running the application

set -e  # Exit on any error

# ANSI color codes for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}    AI Automated CT Scan Radiology Report Generator      ${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"  # Change to the script directory

# Function to print messages
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

# Function to check if conda is installed
check_conda() {
    print_status "Skipping Conda check as the studio provides a default environment."
    return 0
}

# Function to check if environment exists
check_env() {
    print_status "Skipping Conda environment check as the studio provides a default environment."
    return 0
}

# Function to create or update environment
setup_environment() {
    print_status "Skipping Conda environment setup as the studio provides a default environment."
    return 0
}

# Function to run the system requirements check
run_requirements_check() {
    print_status "Running system requirements check..."
    
    if [ ! -f "check_requirements.py" ]; then
        print_error "check_requirements.py not found!"
        return 1
    fi
    
    # Replace with the correct Python interpreter path
    /home/zeus/miniconda3/envs/cloudspace/bin/python check_requirements.py
    
    # Check exit code
    if [ $? -ne 0 ]; then
        print_error "System does not meet all requirements."
        return 1
    else
        print_success "System meets requirements."
        return 0
    fi
}

# Function to display help
show_help() {
    echo -e "Usage: $0 [OPTION] [INPUT_FILE]"
    echo
    echo -e "Options:"
    echo -e "  -h, --help                 Show this help message"
    echo -e "  -c, --check                Run system requirements check"
    echo -e "  -s, --setup                Setup environment and download models"
    echo -e "  -w, --webapp               Start the web application (Streamlit)"
    echo -e "  -a, --api                  Start the API server"
    echo -e "  -p, --process FILE         Process a CT scan file (.nii.gz)"
    echo
    echo -e "Examples:"
    echo -e "  $0 --check                 Check if system meets requirements"
    echo -e "  $0 --setup                 Setup the environment"
    echo -e "  $0 --webapp                Start the web application"
    echo -e "  $0 --process scan.nii.gz   Process a scan and generate a report"
    echo
}

# Function to start web application
start_webapp() {
    print_status "Starting web application..."
    
    # Check if streamlit is installed
    if ! command -v streamlit >/dev/null 2>&1; then
        print_error "Streamlit is not installed or not in PATH."
        return 1
    fi
    
    # Check if app.py exists
    if [ ! -f "app.py" ]; then
        print_error "app.py not found!"
        return 1
    fi
    
    # Start streamlit
    streamlit run app.py --server.port=9000 --server.maxUploadSize=1024
    return 0
}

# Function to start API server
start_api() {
    print_status "Starting API server..."
    
    # Check if flask is installed
    if ! python -c "import flask" 2>/dev/null; then
        print_error "Flask is not installed in the environment."
        return 1
    fi
    
    # Check if api.py exists
    if [ ! -f "api.py" ]; then
        print_error "api.py not found!"
        return 1
    fi
    
    # Start API server
    python api.py
    return 0
}

# Function to process a CT scan file
process_scan() {
    SCAN_FILE="$1"
    
    if [ -z "$SCAN_FILE" ]; then
        print_error "No scan file specified."
        echo -e "Usage: $0 --process FILE"
        return 1
    fi
    
    # Check if file exists
    if [ ! -f "$SCAN_FILE" ]; then
        print_error "File not found: $SCAN_FILE"
        return 1
    fi
    
    # Check file extension
    if [[ "$SCAN_FILE" != *.nii.gz ]]; then
        print_error "File must be in .nii.gz format: $SCAN_FILE"
        return 1
    fi
    
    print_status "Processing CT scan: $SCAN_FILE"
    
    # Check if pipeline.py exists
    if [ ! -f "pipeline.py" ]; then
        print_error "pipeline.py not found!"
        return 1
    fi
    
    # Run the pipeline
    OUTPUT_FILE="${SCAN_FILE%.nii.gz}_report.txt"
    python pipeline.py "$SCAN_FILE" --output "$OUTPUT_FILE"
    
    # Check exit code
    if [ $? -ne 0 ]; then
        print_error "Error processing scan."
        return 1
    else
        print_success "Scan processed successfully."
        print_success "Report saved to: $OUTPUT_FILE"
    fi
    
    return 0
}

# Function to setup models
setup_models() {
    print_status "Setting up models..."
    
    # Create directories
    mkdir -p models/CT-CHAT
    mkdir -p CT_CLIP_encoder
    mkdir -p embeddings
    mkdir -p reports
    
    # Check if encoder model exists
    ENCODER_PATH="CT_CLIP_encoder/clip_visual_encoder.pt"
    if [ ! -f "$ENCODER_PATH" ]; then
        print_warning "CT-CLIP encoder model not found at $ENCODER_PATH"
        print_status "You need to download the encoder model manually."
        print_status "Please place the file at: $SCRIPT_DIR/$ENCODER_PATH"
    else
        print_success "Encoder model found at $ENCODER_PATH"
    fi
    
    # Check if LLM model exists
    LLM_PATH="models/CT-CHAT/llama_3.1_8b"
    if [ ! -d "$LLM_PATH" ]; then
        print_warning "Llama 3.1 8B model not found at $LLM_PATH"
        print_status "You need to download the model from Hugging Face."
        print_status "You'll need proper access credentials for meta-llama models."
        print_status "Directory for model: $SCRIPT_DIR/$LLM_PATH"
        
        echo "Would you like to try downloading it now? (y/n)"
        read -r answer
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            # Check if huggingface_hub is installed
            if ! python -c "import huggingface_hub" 2>/dev/null; then
                print_status "Installing huggingface_hub..."
                pip install huggingface_hub
            fi
            
            print_status "You'll need to login to Hugging Face with appropriate access to Meta's Llama models"
            huggingface-cli login
            
            if [ $? -ne 0 ]; then
                print_error "Failed to login to Hugging Face."
                print_error "You'll need to download the model manually."
                return 1
            fi
            
            print_status "Downloading Llama 3.1 8B model. This may take a while..."
            
            # Check VRAM to determine quantization
            if command -v nvidia-smi >/dev/null 2>&1; then
                VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1 | grep -o -E '[0-9]+')
                
                if [ -n "$VRAM" ] && [ "$VRAM" -lt 16000 ]; then
                    print_warning "Limited VRAM detected. Using 4-bit quantization."
                    USE_4BIT=1
                else
                    USE_4BIT=0
                fi
            else
                print_warning "Could not determine GPU memory. Using 4-bit quantization to be safe."
                USE_4BIT=1
            fi
            
            MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
            
            if [ "$USE_4BIT" -eq 1 ]; then
                print_status "Using 4-bit quantization due to GPU memory constraints."
                python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True
)

print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('$MODEL_ID')
tokenizer.save_pretrained('$LLM_PATH')

print('Downloading model with 4-bit quantization...')
model = AutoModelForCausalLM.from_pretrained(
    '$MODEL_ID',
    device_map='auto',
    quantization_config=quantization_config,
    torch_dtype=torch.float16
)
model.save_pretrained('$LLM_PATH')
print('Model downloaded and saved successfully!')
"
            else
                print_status "Downloading full precision model..."
                python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('$MODEL_ID')
tokenizer.save_pretrained('$LLM_PATH')

print('Downloading model...')
model = AutoModelForCausalLM.from_pretrained(
    '$MODEL_ID',
    device_map='auto',
    torch_dtype=torch.float16
)
model.save_pretrained('$LLM_PATH')
print('Model downloaded and saved successfully!')
"
            fi
            
            if [ $? -ne 0 ]; then
                print_error "Failed to download Llama model."
                print_error "You'll need to download it manually."
                return 1
            fi
            
            print_success "Llama 3.1 8B model downloaded successfully."
        else
            print_warning "Skipping Llama model download. You'll need to add it manually later."
        fi
    else
        print_success "Llama 3.1 8B model found at $LLM_PATH"
    fi
    
    print_success "Model setup complete."
    return 0
}

# Main function
main() {
    # No arguments provided, show help
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    # Parse arguments
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--check)
            run_requirements_check
            exit $?
            ;;
        -s|--setup)
            setup_environment
            run_requirements_check
            setup_models
            exit $?
            ;;
        -w|--webapp)
            start_webapp
            exit $?
            ;;
        -a|--api)
            start_api
            exit $?
            ;;
        -p|--process)
            if [ $# -lt 2 ]; then
                print_error "No scan file specified."
                echo -e "Usage: $0 --process FILE"
                exit 1
            fi
            
            process_scan "$2"
            exit $?
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

# Call main function with all arguments
main "$@"
