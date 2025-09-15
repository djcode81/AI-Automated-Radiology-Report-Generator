import streamlit as st
import os
import subprocess
import tempfile
import time
import re

st.set_page_config(page_title="AI Automated Report Generator", layout="wide")
st.title("AI Automated Radiology Report Generator")

def clean_report_locally(text):
    if text.strip() in ["<s>", "</s>", "<s></s>"]:
        return "No valid report content was generated. Please try again."
    
    if "<s>" in text and "</s>" in text:
        parts = text.split("<s>")
        if len(parts) > 1:
            content = parts[1].split("</s>")[0].strip()
            if content:
                return content
    
    if "[INST]" in text and "[/INST]" in text:
        content = text.split("[/INST]", 1)[1].strip()
        
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
                
        if content:
            return content
    
    if len(text.strip()) > 10:
        return text.strip()
    
    return "No valid report content was generated. Please try again."

st.markdown("Upload a CT scan (.nii.gz format) to generate a comprehensive report")
uploaded_file = st.file_uploader("Choose a file", type=["nii.gz"])

if uploaded_file:
    st.write(f"File: {uploaded_file.name} ({uploaded_file.size / (1024*1024):.2f} MB)")
    
    generate_button = st.button("Generate Report", type="primary")
    
    if generate_button:
        progress = st.progress(0)
        status = st.empty()
        
        status.info("Processing CT scan...")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name
        
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            reports_dir = os.path.join(script_dir, "reports")
            embeddings_dir = os.path.join(script_dir, "embeddings")
            os.makedirs(reports_dir, exist_ok=True)
            os.makedirs(embeddings_dir, exist_ok=True)
            
            temp_file_basename = os.path.basename(temp_path)
            base_filename = temp_file_basename.split('.')[0]
            
            embedding_path = os.path.join(embeddings_dir, f"{base_filename}.npz")
            
            status.info("Generating embeddings...")
            progress.progress(25)
            
            encode_script_path = os.path.join(script_dir, "encode_script.py")
            
            encode_cmd = [
                "python", encode_script_path,
                "--path", temp_path, 
                "--slope", "1", 
                "--intercept", "0", 
                "--xy_spacing", "1", 
                "--z_spacing", "1"
            ]
            
            encode_result = subprocess.run(encode_cmd, capture_output=True, text=True)
            
            if encode_result.returncode != 0:
                st.error(f"Error generating embeddings: {encode_result.stderr}")
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                st.stop()
            
            status.info("Generating radiology report...")
            progress.progress(75)
            
            ct_chat_script_path = os.path.join(script_dir, "enhanced_ct_chat.py")
            llm_model_path = os.path.join(script_dir, "models", "CT-CHAT", "llama_3.1_8b")
            adapter_path = os.path.join(script_dir, "models", "CT-CHAT", "llama_3.1_8b")
            
            report_cmd = [
                "python", ct_chat_script_path,
                "--llm-model-path", llm_model_path,
                "--adapter-path", adapter_path,
                "--image-file", embedding_path, 
                "--device", "cuda"
            ]
            
            report_result = subprocess.run(report_cmd, capture_output=True, text=True)
            
            if report_result.returncode != 0:
                st.error(f"Error generating report: {report_result.stderr}")
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                st.stop()
            
            time.sleep(1)
            
            expected_raw_report_file = f"{base_filename}_ct_chat_raw_report.txt"
            expected_raw_report_path = os.path.join(script_dir, expected_raw_report_file)
            
            report_path = os.path.join(reports_dir, f"{base_filename}_report.txt")
            
            report_content = None
            
            if os.path.exists(expected_raw_report_path):
                with open(expected_raw_report_path, 'r') as f:
                    raw_content = f.read()
                report_content = clean_report_locally(raw_content)
            else:
                find_cmd = ["find", script_dir, "-name", f"*{base_filename}*report*.txt"]
                find_result = subprocess.run(find_cmd, capture_output=True, text=True)
                found_files = find_result.stdout.strip().split('\n')
                found_files = [f for f in found_files if f]
                
                if found_files:
                    raw_files = [f for f in found_files if "raw" in f.lower()]
                    file_to_use = raw_files[0] if raw_files else found_files[0]
                    
                    with open(file_to_use, 'r') as f:
                        raw_content = f.read()
                    report_content = clean_report_locally(raw_content)
                else:
                    st.error(f"Report file not found. Searched for: {expected_raw_report_path}")
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    st.stop()
            
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            status.success("Report generated successfully!")
            progress.progress(100)
            
            st.header("Radiology Report")
            st.text_area("Report Content", report_content, height=400)
            
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
            if os.path.exists(temp_path):
                os.unlink(temp_path)
