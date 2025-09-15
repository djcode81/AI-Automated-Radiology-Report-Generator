#!/usr/bin/env python
import argparse
import torch
import numpy as np
import os
import traceback
import re
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel

def clean_report(text):
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

def check_environment():
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)
        
        if vram_gb < 6:
            if input("GPU has insufficient VRAM. Continue anyway? (y/n): ").lower() != 'y':
                sys.exit(1)
    
    return device

def create_model_with_appropriate_quantization(model_path, device="cuda"):
    if device == "cpu":
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=True
            )
            return model
        except Exception as e:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True
            )
            return model
    
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if vram_gb < 16:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                return model
            except Exception as e:
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="auto",
                        quantization_config=quantization_config,
                        trust_remote_code=True
                    )
                    return model
                except Exception as e:
                    pass
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-model-path", type=str, required=True,
                        help="Path to the language model")
    parser.add_argument("--adapter-path", type=str, required=True,
                        help="Path to the CT-CHAT adapter")
    parser.add_argument("--image-file", type=str, required=True,
                        help=".npz file with the CT embeddings")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference on (auto, cuda, cpu)")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug info")
    args = parser.parse_args()

    if not os.path.exists(args.llm_model_path):
        print(f"Error: Model path {args.llm_model_path} does not exist.")
        sys.exit(1)
    
    if not os.path.exists(args.adapter_path):
        print(f"Error: Adapter path {args.adapter_path} does not exist.")
        sys.exit(1)
    
    if not os.path.exists(args.image_file):
        print(f"Error: Image file {args.image_file} does not exist.")
        sys.exit(1)
    
    device = args.device
    if device is None:
        device = check_environment()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)
        
        torch.cuda.empty_cache()
        base_model = create_model_with_appropriate_quantization(args.llm_model_path, device)
        
        adapter_config_path = os.path.join(args.adapter_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            print(f"Warning: No adapter config found at {adapter_config_path}")
            
        model = PeftModel.from_pretrained(
            base_model,
            args.adapter_path,
            is_trainable=False,
            torch_dtype=torch.float16 if device == "cuda" else None,
            device_map="auto"
        )
        
        ct_data = np.load(args.image_file)
        embeddings = ct_data["arr"]
        
        batch_size = embeddings.shape[0]
        embed_dim = embeddings.shape[-1]
        spatial_dims = embeddings.shape[1:-1]
        
        flat_spatial_dim = np.prod(spatial_dims)
        embeddings_flat = embeddings.reshape(batch_size, flat_spatial_dim, embed_dim)
        embeddings_flat = torch.tensor(embeddings_flat, dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
        
        system_prompt = """You are an expert radiologist specializing in CT interpretation. 
Generate a detailed, structured radiology report that follows standard CT reporting formats.

Your report MUST include:
1. Vascular structures assessment including aorta
2. Mediastinal findings including lymph nodes
3. Lung parenchyma assessment covering all lobes
4. Pleural space evaluation
5. Bone findings including degenerative changes
6. Upper abdominal organs if visible (liver, spleen, adrenals)

Be precise about EVERY finding:
- Report mild findings like emphysema, steatosis, hiatal hernias
- Specify exact locations and sizes in mm where possible
- Note the absence of significant findings in each area"""

        user_prompt = """Analyze this chest CT scan and provide a comprehensive structured report with Findings and Impression sections. Include ALL abnormalities, even mild degenerative changes, emphysema, steatosis or small hernias."""
        
        prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        with torch.inference_mode():
            try:
                model_outputs = model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    encoder_hidden_states=embeddings_flat,
                    return_dict=True,
                )
            except Exception as e:
                try:
                    model_outputs = model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        return_dict=True,
                    )
                except Exception as e:
                    traceback.print_exc()
        
        max_new_tokens = 800
        if device == "cpu":
            temperature = 0.1
            top_p = 0.7
            max_new_tokens = 400
        else:
            temperature = 0.3
            top_p = 0.85
        
        with torch.inference_mode():
            try:
                output_ids = model.generate(
                    input_ids=inputs.input_ids,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=1.2,
                    streamer=streamer,
                )
                
                output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                cleaned_output = clean_report(output_text)
                
                raw_report_file = os.path.basename(args.image_file).replace('.npz', '_ct_chat_raw_report.txt')
                with open(raw_report_file, 'w') as f:
                    f.write(output_text)
                
                report_file = os.path.basename(args.image_file).replace('.npz', '_ct_chat_report.txt')
                with open(report_file, 'w') as f:
                    f.write(cleaned_output)
                
            except Exception as e:
                traceback.print_exc()
                
                try:
                    output_ids = model.generate(
                        input_ids=inputs.input_ids,
                        max_new_tokens=200,
                        streamer=streamer,
                    )
                    
                    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    cleaned_output = clean_report(output_text)
                    
                    report_file = os.path.basename(args.image_file).replace('.npz', '_ct_chat_fallback_report.txt')
                    with open(report_file, 'w') as f:
                        f.write(cleaned_output)
                    
                except Exception as e:
                    traceback.print_exc()
    
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    main()
