#!/usr/bin/env python
import os
import subprocess
import argparse
import re
import sys
import time
from pathlib import Path

def clean_and_format_report(text):
    if "[INST]" in text and "[/INST]" in text:
        text = text.split("[/INST]")[1].strip()
    
    text = re.sub(r'<s>|</s>', '', text)
    text = re.sub(r'Expert Radiologists\' Report', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text

def find_script_dir():
    try:
        script_path = os.path.realpath(__file__)
        script_dir = os.path.dirname(script_path)
    except:
        script_dir = os.getcwd()
    
    return script_dir

def process_nifti_to_report(nifti_path, output_report_path=None, device="auto"):
    if not os.path.exists(nifti_path):
        raise FileNotFoundError(f"Input file not found: {nifti_path}")
    
    if not nifti_path.endswith('.nii.gz'):
        raise ValueError(f"Input file must be a .nii.gz file: {nifti_path}")
    
    if output_report_path is None:
        output_report_path = nifti_path.replace('.nii.gz', '_report.txt')
    
    base_filename = os.path.basename(nifti_path).replace('.nii.gz', '')
    
    script_dir = find_script_dir()
    
    encode_script = os.path.join(script_dir, "encode_script.py")
    ct_chat_script = os.path.join(script_dir, "enhanced_ct_chat.py")
    
    if not os.path.exists(encode_script):
        raise FileNotFoundError(f"encode_script.py not found in {script_dir}")
    
    if not os.path.exists(ct_chat_script):
        raise FileNotFoundError(f"enhanced_ct_chat.py not found in {script_dir}")
    
    embeddings_dir = os.path.join(script_dir, "embeddings")
    reports_dir = os.path.join(script_dir, "reports")
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    llm_model_path = os.path.join(script_dir, "models", "CT-CHAT", "llama_3.1_8b")
    adapter_path = os.path.join(script_dir, "models", "CT-CHAT", "llama_3.1_8b")
    
    if not os.path.exists(llm_model_path):
        print(f"Warning: LLM model not found at {llm_model_path}")
        if not input("Do you want to continue anyway? (y/n): ").lower() == 'y':
            sys.exit(1)
    
    embedding_path = os.path.join(embeddings_dir, f"{base_filename}.npz")
    
    encode_cmd = [
        sys.executable, encode_script,
        '--path', nifti_path,
        '--slope', '1',
        '--intercept', '0',
        '--xy_spacing', '1',
        '--z_spacing', '1'
    ]
    
    try:
        encode_process = subprocess.run(encode_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Failed to encode CT scan") from e
    
    if not os.path.exists(embedding_path):
        alt_embedding_path = os.path.join(os.getcwd(), f"{base_filename}.npz")
        if os.path.exists(alt_embedding_path):
            embedding_path = alt_embedding_path
        else:
            raise FileNotFoundError(f"Embeddings file not found at {embedding_path} or {alt_embedding_path}")
    
    ct_chat_cmd = [
        sys.executable, ct_chat_script,
        '--llm-model-path', llm_model_path,
        '--adapter-path', adapter_path,
        '--image-file', embedding_path,
        '--device', device
    ]
    
    try:
        ct_chat_process = subprocess.run(ct_chat_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Failed to generate report") from e
    
    report_path = embedding_path.replace('.npz', '_ct_chat_report.txt')
    alt_report_path1 = os.path.join(os.getcwd(), f"{base_filename}_ct_chat_report.txt")
    alt_report_path2 = os.path.join(reports_dir, f"{base_filename}_report.txt")
    
    possible_report_paths = [
        report_path,
        alt_report_path1,
        alt_report_path2,
        embedding_path.replace('.npz', '_ct_chat_fallback_report.txt'),
        os.path.join(os.getcwd(), f"{base_filename}_ct_chat_fallback_report.txt")
    ]
    
    report_content = None
    found_report_path = None
    
    for path in possible_report_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    report_content = f.read()
                found_report_path = path
                break
            except Exception as e:
                continue
    
    if report_content is None:
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if base_filename in file and ('report' in file.lower() or 'ct_chat' in file.lower()) and file.endswith('.txt'):
                    try:
                        report_path = os.path.join(root, file)
                        with open(report_path, 'r') as f:
                            report_content = f.read()
                        found_report_path = report_path
                        break
                    except Exception as e:
                        continue
            if report_content is not None:
                break
    
    if report_content is None:
        raise FileNotFoundError("No report file was found. Report generation may have failed.")
    
    cleaned_report = clean_and_format_report(report_content)
    
    with open(output_report_path, 'w') as f:
        f.write(cleaned_report)
    
    return cleaned_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a CT scan to generate a report")
    parser.add_argument("input", help="Path to input .nii.gz file")
    parser.add_argument("--output", help="Path to output report file (optional)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], 
                        help="Device to use for inference (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        sys.exit(1)
    
    if not args.input.endswith('.nii.gz'):
        print(f"Error: Input file must be a .nii.gz file")
        sys.exit(1)
    
    try:
        report = process_nifti_to_report(args.input, args.output, args.device)
        print("Processing complete!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
