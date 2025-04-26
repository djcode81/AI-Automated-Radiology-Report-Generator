#!/usr/bin/env python
"""
System Requirements Checker for CT Scan Radiology Report Generator

This script checks if the current system meets the requirements for running
the CT Scan Radiology Report Generator.
"""

import os
import sys
import platform
import subprocess
import importlib.util
import shutil
from pathlib import Path

# ANSI color codes for better readability
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(message):
    print(f"{Colors.HEADER}{Colors.BOLD}{message}{Colors.ENDC}")

def print_info(message):
    print(f"{Colors.BLUE}[INFO]{Colors.ENDC} {message}")

def print_success(message):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.ENDC} {message}")

def print_warning(message):
    print(f"{Colors.YELLOW}[WARNING]{Colors.ENDC} {message}")

def print_error(message):
    print(f"{Colors.RED}[ERROR]{Colors.ENDC} {message}")

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    print_info("Checking Python version...")
    
    major, minor, _ = platform.python_version_tuple()
    version_str = platform.python_version()
    
    if int(major) < 3 or (int(major) == 3 and int(minor) < 8):
        print_error(f"Python 3.8 or higher is required. Found: {version_str}")
        return False
    else:
        print_success(f"Python version: {version_str}")
        return True

def check_cuda():
    """Check if CUDA is available and get GPU information"""
    print_info("Checking CUDA and GPU availability...")
    
    # Check if torch is installed
    if not importlib.util.find_spec("torch"):
        print_warning("PyTorch is not installed. Cannot check CUDA availability.")
        print_warning("Please install PyTorch with CUDA support.")
        return False
    
    import torch
    
    if not torch.cuda.is_available():
        print_error("CUDA is not available. GPU acceleration will not work.")
        print_info("This could be due to:")
        print_info("  - Missing NVIDIA drivers")
        print_info("  - Incompatible CUDA version")
        print_info("  - PyTorch installed without CUDA support")
        return False
    
    # Get CUDA version
    cuda_version = torch.version.cuda
    print_success(f"CUDA version: {cuda_version}")
    
    # Get GPU count
    device_count = torch.cuda.device_count()
    print_success(f"Number of GPUs: {device_count}")
    
    # Get GPU info for each device
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        device_props = torch.cuda.get_device_properties(i)
        vram_gb = device_props.total_memory / (1024**3)
        
        print_success(f"GPU {i}: {device_name}")
        print_success(f"  Memory: {vram_gb:.2f} GB")
        print_success(f"  CUDA Capability: {device_props.major}.{device_props.minor}")
        
        # Check if VRAM is sufficient
        if vram_gb < 6:
            print_error(f"  Insufficient VRAM! At least 6GB VRAM is required.")
            print_error(f"  4-bit quantization will be used, but performance may be poor.")
        elif vram_gb < 16:
            print_warning(f"  Limited VRAM! 16GB+ VRAM is recommended for optimal performance.")
            print_warning(f"  4-bit quantization will be used.")
        else:
            print_success(f"  Sufficient VRAM for optimal performance.")
    
    return True

def check_nvidia_smi():
    """Check if nvidia-smi is available and working"""
    print_info("Checking nvidia-smi tool...")
    
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode != 0:
            print_error("nvidia-smi command failed. NVIDIA drivers may not be properly installed.")
            return False
        
        # Parse driver version from nvidia-smi output
        driver_version = None
        for line in result.stdout.splitlines():
            if "Driver Version" in line:
                driver_version = line.split("Driver Version:")[1].strip().split()[0]
                break
        
        if driver_version:
            print_success(f"NVIDIA driver version: {driver_version}")
        else:
            print_warning("Could not determine NVIDIA driver version.")
        
        return True
    except FileNotFoundError:
        print_error("nvidia-smi command not found. NVIDIA drivers may not be installed.")
        return False

def check_disk_space():
    """Check if there's enough disk space"""
    print_info("Checking available disk space...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get disk space in GB
    total, used, free = shutil.disk_usage(script_dir)
    total_gb = total / (1024**3)
    free_gb = free / (1024**3)
    
    print_success(f"Total disk space: {total_gb:.1f} GB")
    print_success(f"Free disk space: {free_gb:.1f} GB")
    
    if free_gb < 20:
        print_warning(f"Low disk space! At least 20GB free space is recommended.")
        print_warning(f"Available: {free_gb:.1f} GB")
        return False
    
    return True

def check_system_memory():
    """Check available system memory"""
    print_info("Checking system memory...")
    
    try:
        if platform.system() == "Linux":
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        memory_kb = int(line.split()[1])
                        memory_gb = memory_kb / (1024**2)
                        break
        elif platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulong
            
            class MEMORYSTATUS(ctypes.Structure):
                _fields_ = [
                    ('dwLength', c_ulong),
                    ('dwMemoryLoad', c_ulong),
                    ('dwTotalPhys', c_ulong),
                    ('dwAvailPhys', c_ulong),
                    ('dwTotalPageFile', c_ulong),
                    ('dwAvailPageFile', c_ulong),
                    ('dwTotalVirtual', c_ulong),
                    ('dwAvailVirtual', c_ulong)
                ]
            
            memory_status = MEMORYSTATUS()
            memory_status.dwLength = ctypes.sizeof(MEMORYSTATUS)
            kernel32.GlobalMemoryStatus(ctypes.byref(memory_status))
            memory_gb = memory_status.dwTotalPhys / (1024**3)
        elif platform.system() == "Darwin":  # macOS
            result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            memory_bytes = int(result.stdout.strip())
            memory_gb = memory_bytes / (1024**3)
        else:
            print_warning(f"Unsupported platform: {platform.system()}")
            print_warning("Could not determine system memory.")
            return False
        
        print_success(f"System memory: {memory_gb:.1f} GB")
        
        if memory_gb < 8:
            print_error(f"Insufficient system memory! At least 8GB RAM is required.")
            print_error(f"Available: {memory_gb:.1f} GB")
            return False
        elif memory_gb < 16:
            print_warning(f"Limited system memory! 16GB+ RAM is recommended for optimal performance.")
            print_warning(f"Available: {memory_gb:.1f} GB")
            return True
        else:
            print_success(f"Sufficient system memory for optimal performance.")
            return True
    except Exception as e:
        print_warning(f"Error checking system memory: {e}")
        print_warning("Could not determine system memory.")
        return False

def check_library_dependencies():
    """Check if required Python libraries are installed"""
    print_info("Checking Python library dependencies...")
    
    required_libraries = [
        "torch", "transformers", "peft", "accelerate", "nibabel", 
        "numpy", "streamlit", "flask", "scikit-image"
    ]
    
    missing_libraries = []
    for lib in required_libraries:
        if not importlib.util.find_spec(lib):
            missing_libraries.append(lib)
    
    if missing_libraries:
        print_error(f"Missing required libraries: {', '.join(missing_libraries)}")
        print_info("Please install the missing libraries with pip:")
        print_info(f"pip install {' '.join(missing_libraries)}")
        return False
    else:
        print_success("All required Python libraries are installed.")
        return True

def check_huggingface_credentials():
    """Check if user has Hugging Face credentials configured"""
    print_info("Checking Hugging Face credentials...")
    
    # Check if huggingface_hub is installed
    if not importlib.util.find_spec("huggingface_hub"):
        print_warning("huggingface_hub library is not installed.")
        print_warning("You may need to install it for model downloading:")
        print_warning("pip install huggingface_hub")
        return False
    
    from huggingface_hub import HfApi, HfFolder
    
    # Check if token exists
    token = HfFolder.get_token()
    if token is None:
        print_warning("No Hugging Face token found.")
        print_warning("You may need to log in to access Meta's models:")
        print_warning("huggingface-cli login")
        return False
    
    # Try to validate token
    try:
        api = HfApi()
        user_info = api.whoami(token=token)
        print_success(f"Logged in to Hugging Face as: {user_info['name']}")
        return True
    except Exception as e:
        print_warning(f"Error validating Hugging Face token: {e}")
        print_warning("Your token may be invalid or expired.")
        print_warning("Please log in again: huggingface-cli login")
        return False

def check_model_availability():
    """Check if required models are available or can be downloaded"""
    print_info("Checking model availability...")
    
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Check encoder model
    encoder_path = script_dir / "CT_CLIP_encoder" / "clip_visual_encoder.pt"
    encoder_dir = script_dir / "CT_CLIP_encoder"
    
    if not encoder_dir.exists():
        encoder_dir.mkdir(parents=True, exist_ok=True)
    
    if not encoder_path.exists():
        print_warning(f"CT-CLIP encoder model not found at {encoder_path}")
        print_warning("You will need to obtain this file separately.")
    else:
        print_success(f"CT-CLIP encoder model found at {encoder_path}")
    
    # Check LLM model
    llm_path = script_dir / "models" / "CT-CHAT" / "llama_3.1_8b"
    llm_dir = script_dir / "models" / "CT-CHAT"
    
    if not llm_dir.exists():
        llm_dir.mkdir(parents=True, exist_ok=True)
    
    if not llm_path.exists():
        print_warning(f"Llama 3.1 8B model not found at {llm_path}")
        print_warning("You will need to download this model from Hugging Face.")
        print_warning("Make sure you have access to Meta's Llama models.")
        
        # Check if we can access the model on HuggingFace
        if importlib.util.find_spec("huggingface_hub"):
            from huggingface_hub import HfApi, model_info
            
            try:
                model_id = "meta-llama/Llama-3.1-8B-Instruct"
                info = model_info(model_id)
                print_success(f"Model '{model_id}' is available on Hugging Face.")
                print_info(f"You can download it with the setup.sh script.")
            except Exception as e:
                print_warning(f"Could not verify model availability: {e}")
                print_warning("You may not have access to Meta's models.")
    else:
        # Check if the model files are there
        tokenizer_file = llm_path / "tokenizer_config.json"
        model_file = llm_path / "pytorch_model.bin"
        config_file = llm_path / "config.json"
        
        if tokenizer_file.exists() and (model_file.exists() or (llm_path / "model.safetensors").exists()) and config_file.exists():
            print_success(f"Llama 3.1 8B model found at {llm_path}")
        else:
            print_warning(f"Llama 3.1 8B model directory exists but may be incomplete at {llm_path}")
            print_warning("Please make sure all model files are downloaded correctly.")
    
    return True

def run_basic_inference_test():
    """Run a basic inference test to verify the setup"""
    print_info("Running a basic inference test...")
    
    # Check if we have PyTorch and CUDA
    if not importlib.util.find_spec("torch"):
        print_warning("PyTorch is not installed. Skipping inference test.")
        return False
    
    import torch
    
    if not torch.cuda.is_available():
        print_warning("CUDA is not available. Skipping inference test.")
        return False
    
    try:
        # Create a small tensor and perform a basic operation
        print_info("Testing PyTorch CUDA setup...")
        x = torch.ones(5, 3, device="cuda")
        y = x + 2
        # Get device information
        device_name = torch.cuda.get_device_name(0)
        print_success(f"Basic PyTorch CUDA test passed on {device_name}")
        
        # Try to load a tiny model if transformers is available
        if importlib.util.find_spec("transformers"):
            from transformers import AutoTokenizer, AutoModel
            
            print_info("Testing Transformers library with a small model...")
            tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
            model = AutoModel.from_pretrained("prajjwal1/bert-tiny").to("cuda")
            
            inputs = tokenizer("Hello world!", return_tensors="pt").to("cuda")
            outputs = model(**inputs)
            
            print_success("Successfully loaded and ran a small Transformers model.")
        
        return True
    except Exception as e:
        print_error(f"Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_summary(results):
    """Print a summary of the system check results"""
    print_header("\n=== System Check Summary ===")
    
    all_passed = True
    for check_name, passed in results.items():
        if passed:
            print_success(f"{check_name}: Passed")
        else:
            print_error(f"{check_name}: Failed")
            all_passed = False
    
    if all_passed:
        print_header("\n✅ Your system meets all requirements for running the CT Scan Radiology Report Generator!")
        print_info("You can proceed with the setup and installation.")
    else:
        print_header("\n⚠️ Your system does not meet all requirements.")
        print_info("Please address the issues mentioned above before proceeding.")
        print_info("Some features may not work correctly on your system.")

def main():
    """Main function to run all checks"""
    print_header("CT Scan Radiology Report Generator - System Requirements Check")
    print_header("===========================================================")
    
    results = {}
    
    # Run all checks
    results["Python version"] = check_python_version()
    
    # GPU checks
    try:
        results["CUDA and GPU"] = check_cuda()
    except ImportError:
        print_warning("PyTorch not installed. Skipping CUDA check.")
        results["CUDA and GPU"] = False
    
    results["NVIDIA drivers"] = check_nvidia_smi()
    results["System memory"] = check_system_memory()
    results["Disk space"] = check_disk_space()
    results["Library dependencies"] = check_library_dependencies()
    
    # Optional checks
    try:
        huggingface_check = check_huggingface_credentials()
        results["Hugging Face credentials"] = huggingface_check
    except ImportError:
        print_warning("huggingface_hub not installed. Skipping credential check.")
        results["Hugging Face credentials"] = False
    
    results["Model availability"] = check_model_availability()
    
    # Run inference test only if previous GPU checks passed
    if results.get("CUDA and GPU", False) and results.get("Library dependencies", False):
        results["Inference test"] = run_basic_inference_test()
    else:
        print_warning("Skipping inference test due to failed GPU or library checks.")
        results["Inference test"] = False
    
    # Print summary
    print_summary(results)
    
    # Return overall status for script usage
    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
