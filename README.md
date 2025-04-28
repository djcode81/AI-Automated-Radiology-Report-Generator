# AI Automated CT Scan Radiology Report Generator

This repository contains an AI system for 3D medical image analysis and interpretation that combines a visual encoder and a LoRA-adapted Llama 3.1 LLM model. The system analyzes CT scan images (.nii.gz format) and generates structured radiology reports with findings and impression sections.

## Features

- **3D Medical Image Processing**: Converts CT scan images into embeddings using a custom CT-CLIP encoder
- **AI-Powered Report Generation**: Uses a fine-tuned Llama 3.1 8B model to generate comprehensive radiology reports
- **GPU-Accelerated Workflow**: Optimized tensor operations for faster processing
- **User-Friendly Interface**: Web interface built with Streamlit for easy upload and report generation
- **Multiple Access Methods**: Use via web interface, API, or command line
- **Automatic System Requirements Check**: Verifies your system meets all requirements before running


## How It Works

The system operates in two main stages:

1. **Image Encoding**: 
   - The CT scan is loaded and preprocessed (resampling, normalization)
   - A 3D Vision Transformer encodes the scan into a dense representation
   - The embedding captures the relevant anatomical features

2. **Report Generation**:
   - The CT embeddings are passed to a Llama 3.1 8B model with a LoRA adapter
   - The model generates a structured radiology report with findings and impressions
   - The report is cleaned to remove any artifacts or unwanted content

## Model Details

This system uses two main models:

1. **CT-CLIP Visual Encoder**: Converts 3D CT scans into embeddings
   - Based on a 3D Vision Transformer (ViT) architecture
   - Trained on CT scan data to extract relevant features
   - Located in `CT_CLIP_encoder/clip_visual_encoder.pt`

2. **Llama 3.1 8B with LoRA Adapter**: Generates radiology reports
   - Base: Meta's Llama 3.1 8B model
   - Fine-tuned with LoRA (Low-Rank Adaptation) on radiology reports
   - Located in `models/CT-CHAT/llama_3.1_8b/`
   - Capable of generating structured medical reports with findings and impressions
  
- The models can be downloaded from the [Huggingface repository](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) or from [Google Drive](https://drive.google.com/drive/folders/1NWLCLQoYRIde6e9ht55U2lMXb1b1C2kY?usp=sharing).

## Directory Structure

```
ct-report-generator/
├── run.sh                  # All-in-one script for running the application
├── check_requirements.py   # System requirements checker
├── app.py                  # Streamlit web interface
├── api.py                  # Flask API server
├── pipeline.py             # End-to-end pipeline script
├── encode_script.py        # CT scan to embedding converter
├── enhanced_ct_chat.py     # Report generation script
├── environment.yml         # Conda environment specification
├── embeddings/             # Directory for temporary embeddings
└── reports/                # Directory for generated reports
```



## System Requirements

### Recommended Hardware

- **GPU**: NVIDIA GPU with at least 6GB VRAM (16GB+ recommended)
- **CPU**: Modern multi-core processor
- **RAM**: 16GB+ system memory
- **Storage**: 20GB+ free space for models and temporary files

### Software Requirements

- **Operating System**: Linux (recommended) or Windows with WSL
- **CUDA**: CUDA 11.7+ and compatible NVIDIA drivers
- **Python**: Python 3.8 or higher
- **Conda**: Miniconda or Anaconda for environment management

## Installation

### Quick Setup (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/djcode81/AI-Automated-Radiology-Report.git
   cd AI-Automated-Radiology-Report
   ```

2. Run the all-in-one setup script:
   ```bash
   chmod +x run.sh
   ./run.sh --setup
   ```

The setup script will:
- Check your system for compatibility
- Create a conda environment with required dependencies
- Download necessary models (if authorized)
- Set up the directory structure

### Manual Installation

If the automatic setup doesn't work for your environment, follow these steps:

1. Create a conda environment:
   ```bash
   conda create -n ct-report-env python=3.10
   conda activate ct-report-env
   ```

2. Install PyTorch with CUDA:
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

3. Install other dependencies:
   ```bash
   pip install transformers==4.30.2 peft accelerate safetensors huggingface_hub
   pip install nibabel numpy streamlit flask scikit-image
   ```

4. Create directory structure:
   ```bash
   mkdir -p models/CT-CHAT
   mkdir -p reports
   mkdir -p embeddings
   mkdir -p CT_CLIP_encoder
   ```

5. Download required models, you can either use direct download from Google drive or the Offical Huggingface repository. Mkase sure to place them inside the right folder.

## Usage

### Quick Start Guide

Our all-in-one script provides an easy way to use all features:

```bash
# Check if your system meets the requirements
./run.sh --check

# Set up the environment and download models
./run.sh --setup

# Start the web application
./run.sh --webapp

# Start the API server
./run.sh --api

# Process a CT scan file
./run.sh --process input_file.nii.gz

# Show help information
./run.sh --help
```

### Web Interface

1. Start the Streamlit app:
   ```bash
   ./run.sh --webapp
   ```
   or
   ```bash
   conda activate ct-report-env
   streamlit run app.py --server.port=9000 --server.maxUploadSize=1024
   ```

2. Open your browser and navigate to `http://localhost:9000`

3. Upload a CT scan (.nii.gz format) and click "Generate Report"

### Command Line

Process a CT scan and generate a report:
```bash
./run.sh --process input_file.nii.gz
```

Or directly using Python:
```bash
conda activate ct-report-env
python pipeline.py input_file.nii.gz --output report.txt --device cuda
```

### API

Start the API server:
```bash
./run.sh --api
```
or
```bash
conda activate ct-report-env
python api.py
```

The API will be available at `http://localhost:8000`

Example API usage:
```bash
curl -X POST -F "file=@input_file.nii.gz" http://localhost:8000/generate-report
```


## Troubleshooting

### System Requirements Check

Run the system requirements check to identify any issues:
```bash
./run.sh --check
```

### Common Issues

1. **CUDA out of memory errors**:
   - The system will automatically use 4-bit quantization on GPUs with less than 16GB VRAM
   - Monitor GPU memory usage with `nvidia-smi`
   - Try closing other applications that might be using GPU memory
