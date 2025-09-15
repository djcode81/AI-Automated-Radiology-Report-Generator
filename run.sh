#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

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

check_conda() {
    if command -v conda >/dev/null 2>&1; then
        return 0
    else
        print_error "Conda is not installed or not in PATH."
        return 1
    fi
}

check_env() {
    ENV_NAME="ct-report-env"
    if conda env list | grep -q "^$ENV_NAME "; then
        return 0
    else
        return 1
    fi
}

setup_environment() {
    ENV_NAME="ct-report-env"
    
    if conda env list | grep -q "^$ENV_NAME "; then
        if [ ! -f "environment.yml" ]; then
            create_environment_file
        fi
        conda env update -f environment.yml
    else
        if [ ! -f "environment.yml" ]; then
            create_environment_file
        fi
        conda env create -f environment.yml
    fi
}

create_environment_file() {
    cat > environment.yml << EOF
name: ct-report-env
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda=11.8
  - nibabel
  - numpy
  - streamlit
  - flask
  - pip:
    - transformers>=4.30.0
    - peft
    - accelerate
    - safetensors
    - huggingface_hub
    - streamlit-option-menu
    - scikit-image
EOF
}

run_requirements_check() {
    if [ ! -f "check_requirements.py" ]; then
        print_error "check_requirements.py not found!"
        return 1
    fi
    
    if check_env; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate ct-report-env
    fi
    
    python check_requirements.py
    return $?
}

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

start_webapp() {
    if ! command -v streamlit >/dev/null 2>&1; then
        print_error "Streamlit is not installed or not in PATH."
        return 1
    fi
    
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ct-report-env
    
    if [ ! -f "app.py" ]; then
        print_error "app.py not found!"
        return 1
    fi
    
    streamlit run app.py --server.port=9000 --server.maxUploadSize=1024
    return 0
}

start_api() {
    if ! conda run -n ct-report-env python -c "import flask" 2>/dev/null; then
        print_error "Flask is not installed in the environment."
        return 1
    fi
    
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ct-report-env
    
    if [ ! -f "api.py" ]; then
        print_error "api.py not found!"
        return 1
    fi
    
    python api.py
    return 0
}

process_scan() {
    SCAN_FILE="$1"
    
    if [ -z "$SCAN_FILE" ]; then
        print_error "No scan file specified."
        echo -e "Usage: $0 --process FILE"
        return 1
    fi
    
    if [ ! -f "$SCAN_FILE" ]; then
        print_error "File not found: $SCAN_FILE"
        return 1
    fi
    
    if [[ "$SCAN_FILE" != *.nii.gz ]]; then
        print_error "File must be in .nii.gz format: $SCAN_FILE"
        return 1
    fi
    
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ct-report-env
    
    if [ ! -f "pipeline.py" ]; then
        print_error "pipeline.py not found!"
        return 1
    fi
    
    OUTPUT_FILE="${SCAN_FILE%.nii.gz}_report.txt"
    python pipeline.py "$SCAN_FILE" --output "$OUTPUT_FILE"
    
    if [ $? -ne 0 ]; then
        print_error "Error processing scan."
        return 1
    else
        print_success "Report saved to: $OUTPUT_FILE"
    fi
    
    return 0
}

setup_models() {
    mkdir -p models/CT-CHAT
    mkdir -p CT_CLIP_encoder
    mkdir -p embeddings
    mkdir -p reports
    
    ENCODER_PATH="CT_CLIP_encoder/clip_visual_encoder.pt"
    if [ ! -f "$ENCODER_PATH" ]; then
        print_warning "CT-CLIP encoder model not found at $ENCODER_PATH"
    fi
    
    LLM_PATH="models/CT-CHAT/llama_3.1_8b"
    if [ ! -d "$LLM_PATH" ]; then
        print_warning "Llama 3.1 8B model not found at $LLM_PATH"
        
        echo "Would you like to try downloading it now? (y/n)"
        read -r answer
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            source "$(conda info --base)/etc/profile.d/conda.sh"
            conda activate ct-report-env
            
            if ! python -c "import huggingface_hub" 2>/dev/null; then
                pip install huggingface_hub
            fi
            
            huggingface-cli login
            
            if [ $? -ne 0 ]; then
                print_error "Failed to login to Hugging Face."
                return 1
            fi
            
            if command -v nvidia-smi >/dev/null 2>&1; then
                VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1 | grep -o -E '[0-9]+')
                if [ -n "$VRAM" ] && [ "$VRAM" -lt 16000 ]; then
                    USE_4BIT=1
                else
                    USE_4BIT=0
                fi
            else
                USE_4BIT=1
            fi
            
            MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
            
            if [ "$USE_4BIT" -eq 1 ]; then
                python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained('$MODEL_ID')
tokenizer.save_pretrained('$LLM_PATH')

model = AutoModelForCausalLM.from_pretrained(
    '$MODEL_ID',
    device_map='auto',
    quantization_config=quantization_config,
    torch_dtype=torch.float16
)
model.save_pretrained('$LLM_PATH')
"
            else
                python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('$MODEL_ID')
tokenizer.save_pretrained('$LLM_PATH')

model = AutoModelForCausalLM.from_pretrained(
    '$MODEL_ID',
    device_map='auto',
    torch_dtype=torch.float16
)
model.save_pretrained('$LLM_PATH')
"
            fi
            
            if [ $? -ne 0 ]; then
                print_error "Failed to download Llama model."
                return 1
            fi
        fi
    fi
    
    return 0
}

main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
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
            if ! check_conda; then
                exit 1
            fi
            
            setup_environment
            run_requirements_check
            setup_models
            exit $?
            ;;
        -w|--webapp)
            if ! check_conda || ! check_env; then
                print_error "Environment not setup. Please run setup first: $0 --setup"
                exit 1
            fi
            
            start_webapp
            exit $?
            ;;
        -a|--api)
            if ! check_conda || ! check_env; then
                print_error "Environment not setup. Please run setup first: $0 --setup"
                exit 1
            fi
            
            start_api
            exit $?
            ;;
        -p|--process)
            if [ $# -lt 2 ]; then
                print_error "No scan file specified."
                echo -e "Usage: $0 --process FILE"
                exit 1
            fi
            
            if ! check_conda || ! check_env; then
                print_error "Environment not setup. Please run setup first: $0 --setup"
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

main "$@"
