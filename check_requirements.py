#!/usr/bin/env python
import os
import sys
import platform
import subprocess
import importlib.util
import shutil
from pathlib import Path

def check_python_version():
    major, minor, _ = platform.python_version_tuple()
    return int(major) >= 3 and (int(major) > 3 or int(minor) >= 8)

def check_cuda():
    if not importlib.util.find_spec("torch"):
        return False
    
    import torch
    if not torch.cuda.is_available():
        return False
    
    device_props = torch.cuda.get_device_properties(0)
    vram_gb = device_props.total_memory / (1024**3)
    return vram_gb >= 6

def check_nvidia_smi():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def check_disk_space():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    total, used, free = shutil.disk_usage(script_dir)
    free_gb = free / (1024**3)
    return free_gb >= 20

def check_system_memory():
    try:
        if platform.system() == "Linux":
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        memory_kb = int(line.split()[1])
                        memory_gb = memory_kb / (1024**2)
                        return memory_gb >= 8
        elif platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulong
            
            class MEMORYSTATUS(ctypes.Structure):
                _fields_ = [('dwLength', c_ulong), ('dwMemoryLoad', c_ulong),
                           ('dwTotalPhys', c_ulong), ('dwAvailPhys', c_ulong),
                           ('dwTotalPageFile', c_ulong), ('dwAvailPageFile', c_ulong),
                           ('dwTotalVirtual', c_ulong), ('dwAvailVirtual', c_ulong)]
            
            memory_status = MEMORYSTATUS()
            memory_status.dwLength = ctypes.sizeof(MEMORYSTATUS)
            kernel32.GlobalMemoryStatus(ctypes.byref(memory_status))
            memory_gb = memory_status.dwTotalPhys / (1024**3)
            return memory_gb >= 8
        elif platform.system() == "Darwin":
            result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            memory_bytes = int(result.stdout.strip())
            memory_gb = memory_bytes / (1024**3)
            return memory_gb >= 8
        return False
    except Exception:
        return False

def check_library_dependencies():
    required_libraries = ["torch", "transformers", "peft", "accelerate", "nibabel", 
                         "numpy", "streamlit", "flask", "scikit-image"]
    
    for lib in required_libraries:
        if not importlib.util.find_spec(lib):
            return False
    return True

def check_huggingface_credentials():
    if not importlib.util.find_spec("huggingface_hub"):
        return False
    
    try:
        from huggingface_hub import HfApi, HfFolder
        token = HfFolder.get_token()
        if token is None:
            return False
        
        api = HfApi()
        api.whoami(token=token)
        return True
    except Exception:
        return False

def check_model_availability():
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    encoder_path = script_dir / "CT_CLIP_encoder" / "clip_visual_encoder.pt"
    llm_path = script_dir / "models" / "CT-CHAT" / "llama_3.1_8b"
    
    encoder_exists = encoder_path.exists()
    llm_exists = llm_path.exists() and (llm_path / "config.json").exists()
    
    return encoder_exists and llm_exists

def run_basic_inference_test():
    if not importlib.util.find_spec("torch"):
        return False
    
    import torch
    if not torch.cuda.is_available():
        return False
    
    try:
        x = torch.ones(5, 3, device="cuda")
        y = x + 2
        return True
    except Exception:
        return False

def main():
    checks = {
        "Python version": check_python_version,
        "CUDA and GPU": check_cuda,
        "NVIDIA drivers": check_nvidia_smi,
        "System memory": check_system_memory,
        "Disk space": check_disk_space,
        "Library dependencies": check_library_dependencies,
        "Hugging Face credentials": check_huggingface_credentials,
        "Model availability": check_model_availability,
        "Inference test": run_basic_inference_test
    }
    
    results = {}
    for name, check_func in checks.items():
        try:
            results[name] = check_func()
        except Exception:
            results[name] = False
    
    all_passed = all(results.values())
    
    if not all_passed:
        failed_checks = [name for name, passed in results.items() if not passed]
        print(f"Requirements check failed: {', '.join(failed_checks)}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
