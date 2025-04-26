#!/usr/bin/env python
# pipeline.py - End-to-end pipeline for CT scan to radiology report generation
import os
import subprocess
import argparse
import re
import sys
import time
from pathlib import Path

def clean_and_format_report(text):
    """Clean the report by removing prompts and keeping the actual content"""
    # Remove the prompt/instruction part
    if "[INST]" in text and "[/INST]" in text:
        text = text.split("[/INST]")[1].strip()
    
    # Remove <s> and </s> tags
    text = re.sub(r'<s>|</s>', '', text)
    
    # Remove any meta-text like "Expert Radiologists' Report"
    text = re.sub(r'Expert Radiologists\' Report', '', text)
    
    # Remove any remaining XML-like tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove extra whitespace and empty lines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text

def find_script_dir():
    """Find the directory where this script is located"""
    # Try to get the real path of the current script
    try:
        script_path = os.path.realpath(__file__)
        script_dir = os.path.dirname(script_path)
    except:
        # If that fails, use the current working directory
        script_dir = os.getcwd()
    
    return script_dir

def process_nifti_to_report(nifti_path, output_report_path=None, device="auto"):
    """Process a .nii.gz file to generate a radiology report
    
    Args:
        nifti_path (str): Path to input .nii.gz file
        output_report_path (str, optional): Path to save the output report
        device (str, optional): Device to use for inference ('auto', 'cuda', 'cpu')
        
    Returns:
        str: The generated radiology report text
    """
    print(f"Processing CT scan: {nifti_path}")
    
    # Check if the input file exists
    if not os.path.exists(nifti_path):
        raise FileNotFoundError(f"Input file not found: {nifti_path}")
    
    # Check if the input file is a .nii.gz file
    if not nifti_path.endswith('.nii.gz'):
        raise ValueError(f"Input file must be a .nii.gz file: {nifti_path}")
    
    # Set default output path if not provided
    if output_report_path is None:
        output_report_path = nifti_path.replace('.nii.gz', '_report.txt')
    
    # Get base filename
    base_filename = os.path.basename(nifti_path).replace('.nii.gz', '')
    
    # Find the directory of this script
    script_dir = find_script_dir()
    
    # Find paths for encode_script.py and enhanced_ct_chat.py
    encode_script = os.path.join(script_dir, "encode_script.py")
    ct_chat_script = os.path.join(script_dir, "enhanced_ct_chat.py")
    
    # Check if the scripts exist
    if not os.path.exists(encode_script):
        encode_script = os.path.join(script_dir, "encode_script.py")
        if not os.path.exists(encode_script):
            raise FileNotFoundError(f"encode_script.py not found in {script_dir}")
    
    if not os.path.exists(ct_chat_script):
        ct_chat_script = os.path.join(script_dir, "enhanced_ct_chat.py")
        if not os.path.exists(ct_chat_script):
            raise FileNotFoundError(f"enhanced_ct_chat.py not found in {script_dir}")
    
    # Make embeddings directory if it doesn't exist
    embeddings_dir = os.path.join(script_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Make reports directory if it doesn't exist
    reports_dir = os.path.join(script_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Set paths for models
    encoder_path = os.path.join(script_dir, "CT_CLIP_encoder", "clip_visual_encoder.pt")
    llm_model_path = os.path.join(script_dir, "models", "CT-CHAT", "llama_3.1_8b")
    adapter_path = os.path.join(script_dir, "models", "CT-CHAT", "llama_3.1_8b")
    
    # Check for encoder model
    if not os.path.exists(encoder_path):
        print(f"Warning: Encoder model not found at {encoder_path}")
        print("Please make sure you've downloaded the encoder model.")
        if not input("Do you want to continue anyway? (y/n): ").lower() == 'y':
            sys.exit(1)
    
    # Check for LLM model
    if not os.path.exists(llm_model_path):
        print(f"Warning: LLM model not found at {llm_model_path}")
        print("Please make sure you've downloaded the LLM model.")
        if not input("Do you want to continue anyway? (y/n): ").lower() == 'y':
            sys.exit(1)
    
    # Set path for the embeddings file
    embedding_path = os.path.join(embeddings_dir, f"{base_filename}.npz")
    
    # Step 1: Run encode_script.py to convert .nii.gz to .npz
    print(f"Step 1: Converting CT scan to embeddings...")
    encode_cmd = [
        sys.executable, encode_script,
        '--path', nifti_path,
        '--slope', '1',
        '--intercept', '0',
        '--xy_spacing', '1',
        '--z_spacing', '1'
    ]
    
    try:
        print(f"Running command: {' '.join(encode_cmd)}")
        encode_process = subprocess.run(encode_cmd, check=True, capture_output=True, text=True)
        print(encode_process.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running encode_script.py: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        raise RuntimeError("Failed to encode CT scan") from e
    
    # Check if embeddings were created
    if not os.path.exists(embedding_path):
        # Look for embeddings in the current directory as fallback
        alt_embedding_path = os.path.join(os.getcwd(), f"{base_filename}.npz")
        if os.path.exists(alt_embedding_path):
            embedding_path = alt_embedding_path
        else:
            raise FileNotFoundError(f"Embeddings file not found at {embedding_path} or {alt_embedding_path}")
    
    # Step 2: Run enhanced_ct_chat.py to generate report
    print(f"Step 2: Generating report from embeddings...")
    ct_chat_cmd = [
        sys.executable, ct_chat_script,
        '--llm-model-path', llm_model_path,
        '--adapter-path', adapter_path,
        '--image-file', embedding_path,
        '--device', device
    ]
    
    try:
        print(f"Running command: {' '.join(ct_chat_cmd)}")
        ct_chat_process = subprocess.run(ct_chat_cmd, check=True, capture_output=True, text=True)
        print(ct_chat_process.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running enhanced_ct_chat.py: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        raise RuntimeError("Failed to generate report") from e
    
    # Look for the report in several possible locations
    report_path = embedding_path.replace('.npz', '_ct_chat_report.txt')
    alt_report_path1 = os.path.join(os.getcwd(), f"{base_filename}_ct_chat_report.txt")
    alt_report_path2 = os.path.join(reports_dir, f"{base_filename}_report.txt")
    
    # Check various paths where the report might be
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
                print(f"Found report at {path}")
                break
            except Exception as e:
                print(f"Error reading report at {path}: {e}")
    
    if report_content is None:
        # Try to find any file that might contain the report
        print("Report not found in expected locations. Searching for report files...")
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if base_filename in file and ('report' in file.lower() or 'ct_chat' in file.lower()) and file.endswith('.txt'):
                    try:
                        report_path = os.path.join(root, file)
                        with open(report_path, 'r') as f:
                            report_content = f.read()
                        found_report_path = report_path
                        print(f"Found report at {report_path}")
                        break
                    except Exception as e:
                        print(f"Error reading report at {report_path}: {e}")
            if report_content is not None:
                break
    
    if report_content is None:
        raise FileNotFoundError("No report file was found. Report generation may have failed.")
    
    # Clean the report
    cleaned_report = clean_and_format_report(report_content)
    
    # Save cleaned report to requested output path
    with open(output_report_path, 'w') as f:
        f.write(cleaned_report)
    
    print(f"Report generated successfully and saved to: {output_report_path}")
    
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
        print("\nReport preview:")
        print("=" * 40)
        print(report[:300] + ("..." if len(report) > 300 else ""))
        print("=" * 40)
        print("Processing complete!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
