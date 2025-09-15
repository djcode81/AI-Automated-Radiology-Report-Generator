#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}    AI Automated CT Scan Radiology Report Generator      ${NC}"
echo -e "${BLUE}                 System Setup Utility                    ${NC}"
echo -e "${BLUE}=========================================================${NC}"

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

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

check_cuda() {
    if ! command_exists nvidia-smi; then
        print_error "NVIDIA drivers are not installed or nvidia-smi is not in the PATH."
        print_error "Please install NVIDIA drivers compatible with CUDA 11.7 or higher."
        return 1
    fi
    
    cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
    gpu_model=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)
    
    vram_mb=$(echo $vram | grep -o -E '[0-9]+')
    
    if [ -z "$vram_mb" ]; then
        print_error "Could not determine GPU memory size."
        return 1
    fi
    
    if [ $vram_mb -lt 6000 ]; then
        print_error "Insufficient GPU memory. At least 6GB VRAM is required for the 4-bit quantized model."
        return 1
    elif [ $vram_mb -lt 16000 ]; then
        print_warning "Your GPU has less than 16GB VRAM ($vram). We'll use 4-bit quantization for the model."
        export USE_4BIT_QUANTIZATION=1
    else
        print_success "Your GPU meets the memory requirements with $vram."
        export USE_4BIT_QUANTIZATION=0
    fi
    
    return 0
}

check_python() {
    if ! command_exists python3; then
        print_error "Python 3 is not installed or not in the PATH."
        print_error "Please install Python 3.8 or higher."
        return 1
    fi
    
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    python_major=$(echo $python_version | cut -d. -f1)
    python_minor=$(echo $python_version | cut -d. -f2)
    
    if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]); then
        print_error "Python version must be at least 3.8. Found $python_version."
        print_error "Please upgrade your Python installation."
        return 1
    fi
    
    print_success "Python version check passed."
    return 0
}

setup_conda_environment() {
    if ! command_exists conda; then
        print_error "Conda is not installed or not in the PATH."
        print_error "Please install Miniconda or Anaconda before proceeding."
        return 1
    fi
    
    ENV_NAME="ct-report-env"
    
    if conda env list | grep -q "^$ENV_NAME "; then
        conda env update -f environment.yml
    else
        if [ ! -f "environment.yml" ]; then
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
        fi
        
        conda env create -f environment.yml
    fi
    
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate $ENV_NAME
    
    if [ $? -ne 0 ]; then
        print_error "Failed to activate conda environment '$ENV_NAME'."
        return 1
    fi
    
    print_success "Conda environment setup complete."
    return 0
}

create_directory_structure() {
    mkdir -p models/CT-CHAT
    mkdir -p reports
    mkdir -p embeddings
    mkdir -p CT_CLIP_encoder
    
    print_success "Directory structure created."
    return 0
}

check_model_files() {
    ENCODER_PATH="CT_CLIP_encoder/clip_visual_encoder.pt"
    if [ ! -f "$ENCODER_PATH" ]; then
        print_warning "CT-CLIP encoder model not found at $ENCODER_PATH"
        print_status "You need to download the encoder model manually."
        print_status "Please place the file at: $SCRIPT_DIR/$ENCODER_PATH"
        echo "Would you like to try downloading it now? (y/n)"
        read -r answer
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            echo "Please obtain the actual model file from the original repository or project authors."
            return 1
        else
            print_warning "Skipping encoder model download. You'll need to add it manually later."
        fi
    else
        print_success "Encoder model found at $ENCODER_PATH"
    fi
    
    LLM_PATH="models/CT-CHAT/llama_3.1_8b"
    if [ ! -d "$LLM_PATH" ]; then
        print_warning "Llama 3.1 8B model not found at $LLM_PATH"
        print_status "You need to download the model from Hugging Face."
        print_status "You'll need proper access credentials for meta-llama models."
        
        echo "Would you like to try downloading it now? (y/n)"
        read -r answer
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            if ! command_exists huggingface-cli; then
                pip install huggingface_hub
            fi
            
            huggingface-cli login
            
            if [ $? -ne 0 ]; then
                print_error "Failed to login to Hugging Face."
                return 1
            fi
            
            MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
            
            if [ "$USE_4BIT_QUANTIZATION" -eq 1 ]; then
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
            
            print_success "Llama 3.1 8B model downloaded successfully."
        else
            print_warning "Skipping Llama model download. You'll need to add it manually later."
        fi
    else
        print_success "Llama 3.1 8B model found at $LLM_PATH"
    fi
    
    return 0
}

generate_run_scripts() {
    cat > run_app.sh << EOF
#!/bin/bash
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct-report-env
cd "\$(dirname "\$0")"
streamlit run app.py --server.port=9000 --server.maxUploadSize=1024
EOF
    chmod +x run_app.sh
    
    cat > run_pipeline.sh << EOF
#!/bin/bash
if [ \$# -lt 1 ]; then
    echo "Usage: \$0 <input_file.nii.gz> [output_report.txt]"
    exit 1
fi

INPUT_FILE="\$1"
OUTPUT_FILE="\${2:-\${INPUT_FILE%.nii.gz}_report.txt}"

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ct-report-env
cd "\$(dirname "\$0")"
python pipeline.py "\$INPUT_FILE" --output "\$OUTPUT_FILE"
EOF
    chmod +x run_pipeline.sh
    
    print_success "Run scripts generated successfully."
    return 0
}

main() {
    if ! check_cuda; then
        print_error "GPU check failed. Cannot proceed with setup."
        exit 1
    fi
    
    if ! check_python; then
        print_error "Python check failed. Cannot proceed with setup."
        exit 1
    fi
    
    if ! setup_conda_environment; then
        print_error "Conda environment setup failed. Cannot proceed."
        exit 1
    fi
    
    if ! create_directory_structure; then
        print_error "Failed to create directory structure."
        exit 1
    fi
    
    if ! check_model_files; then
        print_warning "Some model files could not be automatically set up."
    fi
    
    if ! generate_run_scripts; then
        print_error "Failed to generate run scripts."
        exit 1
    fi
    
    print_success "Setup complete! You can now run the application using:"
    echo -e "${GREEN}  ./run_app.sh${NC} - To start the Streamlit web interface"
    echo -e "${GREEN}  ./run_pipeline.sh <input.nii.gz>${NC} - To process a file from command line"
    
    return 0
}

main
