#!/bin/bash
# CT Scan Radiology Report Generator - System Setup and Verification
# This script checks system requirements, installs dependencies, and sets up the AI system
# for 3D medical image analysis and interpretation

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
echo -e "${BLUE}                 System Setup Utility                    ${NC}"
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

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if CUDA is available and get GPU information
check_cuda() {
    print_status "Checking CUDA and GPU availability..."
    
    if ! command_exists nvidia-smi; then
        print_error "NVIDIA drivers are not installed or nvidia-smi is not in the PATH."
        print_error "Please install NVIDIA drivers compatible with CUDA 11.7 or higher."
        return 1
    fi
    
    # Check CUDA version
    cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
    print_status "NVIDIA Driver version: $cuda_version"
    
    # Get GPU model
    gpu_model=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    print_status "GPU model detected: $gpu_model"
    
    # Get VRAM
    vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)
    print_status "GPU memory: $vram"
    
    # Extract memory size in MB
    vram_mb=$(echo $vram | grep -o -E '[0-9]+')
    
    if [ -z "$vram_mb" ]; then
        print_error "Could not determine GPU memory size."
        return 1
    fi
    
    # Check minimum VRAM requirements (6GB for 4-bit quantized model)
    if [ $vram_mb -lt 6000 ]; then
        print_error "Insufficient GPU memory. At least 6GB VRAM is required for the 4-bit quantized model."
        print_error "Your GPU has approximately $vram_mb MB."
        return 1
    elif [ $vram_mb -lt 16000 ]; then
        print_warning "Your GPU has less than 16GB VRAM ($vram). We'll use 4-bit quantization for the model."
        print_warning "For optimal performance, 16GB or more VRAM is recommended."
        export USE_4BIT_QUANTIZATION=1
    else
        print_success "Your GPU meets the memory requirements with $vram."
        export USE_4BIT_QUANTIZATION=0
    fi
    
    return 0
}

# Function to check Python version
check_python() {
    print_status "Checking Python version..."
    
    if ! command_exists python3; then
        print_error "Python 3 is not installed or not in the PATH."
        print_error "Please install Python 3.8 or higher."
        return 1
    fi
    
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_status "Python version: $python_version"
    
    # Check Python version is at least 3.8
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

# Function to create directory structure
create_directory_structure() {
    print_status "Creating project directory structure..."
    
    # Create directories
    mkdir -p models/CT-CHAT
    mkdir -p reports
    mkdir -p embeddings
    mkdir -p CT_CLIP_encoder
    
    print_success "Directory structure created."
    return 0
}

# Function to check for and download model files
check_model_files() {
    print_status "Checking for required model files..."
    
    # Check for encoder model
    ENCODER_PATH="CT_CLIP_encoder/clip_visual_encoder.pt"
    if [ ! -f "$ENCODER_PATH" ]; then
        print_warning "CT-CLIP encoder model not found at $ENCODER_PATH"
        print_status "You need to download the encoder model manually."
        print_status "Please place the file at: $SCRIPT_DIR/$ENCODER_PATH"
        echo "Would you like to try downloading it now? (y/n)"
        read -r answer
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            print_status "Attempting to download encoder model..."
            # This is a placeholder. In a real scenario, you would provide a direct download link
            # Here we're just creating a placeholder file for demo purposes
            echo "Please obtain the actual model file from the original repository or project authors."
            return 1
        else
            print_warning "Skipping encoder model download. You'll need to add it manually later."
        fi
    else
        print_success "Encoder model found at $ENCODER_PATH"
    fi
    
    # Check for LLM model
    LLM_PATH="models/CT-CHAT/llama_3.1_8b"
    if [ ! -d "$LLM_PATH" ]; then
        print_warning "Llama 3.1 8B model not found at $LLM_PATH"
        print_status "You need to download the model from Hugging Face."
        print_status "You'll need proper access credentials for meta-llama models."
        print_status "Directory for model: $SCRIPT_DIR/$LLM_PATH"
        
        echo "Would you like to try downloading it now? (y/n)"
        read -r answer
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            if ! command_exists huggingface-cli; then
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
            
            # Determine whether to use 4-bit quantization based on available GPU memory
            MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
            
            if [ "$USE_4BIT_QUANTIZATION" -eq 1 ]; then
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
    
    return 0
}

# Function to generate run scripts
generate_run_scripts() {
    print_status "Generating run scripts..."
    
    # Create run_app.sh script for Streamlit
    cat > run_app.sh << EOF
#!/bin/bash
# Script to run the Streamlit app for AI Automated Report Generator

# Run Streamlit app
cd "\$(dirname "\$0")"
streamlit run app.py --server.port=9000 --server.maxUploadSize=1024
EOF
    chmod +x run_app.sh
    
    # Create run_pipeline.sh script for command line usage
    cat > run_pipeline.sh << EOF
#!/bin/bash
# Script to run the CT scan analysis pipeline from command line

# Check if input file was provided
if [ \$# -lt 1 ]; then
    echo "Usage: \$0 <input_file.nii.gz> [output_report.txt]"
    exit 1
fi

INPUT_FILE="\$1"
OUTPUT_FILE="\${2:-\${INPUT_FILE%.nii.gz}_report.txt}"

# Run the pipeline
cd "\$(dirname "\$0")"
python pipeline.py "\$INPUT_FILE" --output "\$OUTPUT_FILE"
EOF
    chmod +x run_pipeline.sh
    
    print_success "Run scripts generated successfully."
    return 0
}

# Main function
main() {
    print_status "Starting system requirements check and setup..."
    
    # Check CUDA and GPU
    if ! check_cuda; then
        print_error "GPU check failed. Cannot proceed with setup."
        exit 1
    fi
    
    # Check Python version
    if ! check_python; then
        print_error "Python check failed. Cannot proceed with setup."
        exit 1
    fi
    
    # Skip Conda environment setup
    print_status "Skipping Conda environment setup as the studio provides a default environment."
    
    # Create directory structure
    if ! create_directory_structure; then
        print_error "Failed to create directory structure."
        exit 1
    fi
    
    # Check model files
    if ! check_model_files; then
        print_warning "Some model files could not be automatically set up."
        print_warning "You may need to manually download and place them in the correct locations."
    fi
    
    # Generate run scripts
    if ! generate_run_scripts; then
        print_error "Failed to generate run scripts."
        exit 1
    fi
    
    print_success "Setup complete! You can now run the application using:"
    echo -e "${GREEN}  ./run_app.sh${NC} - To start the Streamlit web interface"
    echo -e "${GREEN}  ./run_pipeline.sh <input.nii.gz>${NC} - To process a file from command line"
    
    return 0
}

# Run the main function
main
