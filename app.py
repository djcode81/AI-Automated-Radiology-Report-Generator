import streamlit as st
import os
import subprocess
import tempfile
import time
import re

# Set page configuration
st.set_page_config(page_title="AI Automated Report Generator", layout="wide")
# Title
st.title("AI Automated Radiology Report Generator")

def clean_report_locally(text):
    """Clean the report text within app.py without modifying ct_chat_simple.py"""
    # First, check if the text is just <s> or similar
    if text.strip() in ["<s>", "</s>", "<s></s>"]:
        return "No valid report content was generated. Please try again."
    
    # For texts with <s> tags
    if "<s>" in text and "</s>" in text:
        # Extract just the content between <s> and </s>
        parts = text.split("<s>")
        if len(parts) > 1:
            content = parts[1].split("</s>")[0].strip()
            if content:  # If we have content, return it
                return content
    
    # For texts with [INST] tags
    if "[INST]" in text and "[/INST]" in text:
        # Get everything after [/INST]
        content = text.split("[/INST]", 1)[1].strip()
        
        # Remove trailing tags or instruction markers
        end_markers = [
            "</s>", 
            "[INST]", 
            "Please generate", 
            "Please provide",
            "Please answer",
            "Based on the",
            "According to",
            "In conclusion",
            "Please use"
        ]
        
        for marker in end_markers:
            if marker in content:
                content = content.split(marker)[0].strip()
                
        # Also handle instruction patterns
        instruction_patterns = [
            r'\s+Using the information',
            r'\s+Given the context',
            r'\s+From the CT',
            r'\s+Can you provide',
            r'\s+What are the',
            r'\s+Does the patient'
        ]
        
        for pattern in instruction_patterns:
            match = re.search(pattern, content)
            if match:
                content = content[:match.start()].strip()
                
        if content:  # If we extracted content, return it
            return content
    
    # If we didn't match any of the above patterns but there's content, return it
    if len(text.strip()) > 10:  # Arbitrary minimum to avoid empty or tiny fragments
        return text.strip()
    
    # Fallback message for when we can't extract proper content
    return "No valid report content was generated. Please try again."

# File uploader with larger size limit
st.markdown("Upload a CT scan (.nii.gz format) to generate a comprehensive report")
uploaded_file = st.file_uploader("Choose a file", type=["nii.gz"])

if uploaded_file:
    # Display file info
    st.write(f"File: {uploaded_file.name} ({uploaded_file.size / (1024*1024):.2f} MB)")
    
    # Generate button
    generate_button = st.button("Generate Report", type="primary")
    
    if generate_button:
        progress = st.progress(0)
        status = st.empty()
        
        # Save uploaded file to temporary location
        status.info("Processing CT scan...")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name
        
        try:
            # Make sure the reports directory exists
            reports_dir = os.path.expanduser("~/DS/new1/reports")
            embeddings_dir = os.path.expanduser("~/DS/new1/embeddings")
            os.makedirs(reports_dir, exist_ok=True)
            os.makedirs(embeddings_dir, exist_ok=True)
            
            # Get base filename
            temp_file_basename = os.path.basename(temp_path)
            base_filename = temp_file_basename.split('.')[0]
            
            # Path to the embedding file
            embedding_path = os.path.join(embeddings_dir, f"{base_filename}.npz")
            
            # Step 1: Generate embeddings (25%)
            status.info("Generating embeddings...")
            progress.progress(25)
            
            # Run encoding script
            encode_cmd = [
                "bash", "-c",
                f"cd /net/dali/home/mscbio/dhp72/DS/new1 && " +
                f"source ~/miniconda3/etc/profile.d/conda.sh && " +
                f"conda activate clara-env && " +
                f"python /net/dali/home/mscbio/dhp72/DS/new1/encode_script.py " +
                f"--path {temp_path} --slope 1 --intercept 0 --xy_spacing 1 --z_spacing 1"
            ]
            
            encode_result = subprocess.run(encode_cmd, capture_output=True, text=True)
            
            if encode_result.returncode != 0:
                st.error(f"Error generating embeddings: {encode_result.stderr}")
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                st.stop()
            
            # Step 2: Generate report (75%)
            status.info("Generating radiology report...")
            progress.progress(75)
            
            # Run the report generation script in the DS/new1 directory
            report_cmd = [
                "bash", "-c",
                f"cd /net/dali/home/mscbio/dhp72/DS/new1 && " +
                f"source ~/miniconda3/etc/profile.d/conda.sh && " +
                f"conda activate clara-env && " +
                f"python enhanced_ct_chat.py " +
                f"--llm-model-path ~/DS/new1/models/CT-CHAT/llama_3.1_8b " +
                f"--adapter-path ~/DS/new1/models/CT-CHAT/llama_3.1_8b " +
                f"--image-file {embedding_path} --device cuda"
            ]
            
            report_result = subprocess.run(report_cmd, capture_output=True, text=True)
            
            if report_result.returncode != 0:
                st.error(f"Error generating report: {report_result.stderr}")
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                st.stop()
            
            # Wait a moment for file system to update
            time.sleep(1)
            
            # ALWAYS look for the RAW report first
            expected_raw_report_file = f"{base_filename}_ct_chat_raw_report.txt"
            expected_raw_report_path = os.path.join("/net/dali/home/mscbio/dhp72/DS/new1", expected_raw_report_file)
            
            # Copy report to the reports directory
            report_path = os.path.join(reports_dir, f"{base_filename}_report.txt")
            
            report_content = None
            
            # Check for the raw report file first
            if os.path.exists(expected_raw_report_path):
                with open(expected_raw_report_path, 'r') as f:
                    raw_content = f.read()
                # Clean the report content using our local function
                report_content = clean_report_locally(raw_content)
            else:
                # Try finding any report file
                find_cmd = ["find", "/net/dali/home/mscbio/dhp72/DS/new1", "-name", f"*{base_filename}*report*.txt"]
                find_result = subprocess.run(find_cmd, capture_output=True, text=True)
                found_files = find_result.stdout.strip().split('\n')
                found_files = [f for f in found_files if f]
                
                if found_files:
                    # Prioritize raw report if found
                    raw_files = [f for f in found_files if "raw" in f.lower()]
                    file_to_use = raw_files[0] if raw_files else found_files[0]
                    
                    with open(file_to_use, 'r') as f:
                        raw_content = f.read()
                    # Clean the report content using our local function
                    report_content = clean_report_locally(raw_content)
                else:
                    st.error(f"Report file not found. Searched for: {expected_raw_report_path}")
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    st.stop()
            
            # Save the cleaned report
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            status.success("Report generated successfully!")
            progress.progress(100)
            
            # Display the report
            st.header("Radiology Report")
            st.text_area("Report Content", report_content, height=400)
            
            # Download button
            st.download_button(
                label="Download Report",
                data=report_content,
                file_name=f"{uploaded_file.name.replace('.nii.gz', '')}_report.txt",
                mime="text/plain"
            )
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error(f"Error details: {str(e.__class__.__name__)}")
            import traceback
            st.code(traceback.format_exc())
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
