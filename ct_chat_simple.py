#!/usr/bin/env python
import argparse
import torch
import numpy as np
import os
import traceback
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import TextStreamer

def clean_report(text):
    if text.startswith("<s>") and "</s>" in text:
        text = text.split("<s>")[1].split("</s>")[0].strip()
    else:
        end_markers = [
            "</s>", 
            "[INST]", 
            "[/INST]", 
            "Please generate", 
            "Please provide",
            "Please answer",
            "Based on the",
            "According to",
            "In conclusion",
            "Please use"
        ]
        
        for marker in end_markers:
            if marker in text:
                text = text.split(marker)[0]
        
        instruction_patterns = [
            r'\s+Using the information',
            r'\s+Given the context',
            r'\s+From the CT',
            r'\s+Can you provide',
            r'\s+What are the',
            r'\s+Does the patient'
        ]
        
        for pattern in instruction_patterns:
            match = re.search(pattern, text)
            if match:
                text = text[:match.start()]
    
    text = text.strip()
    
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-model-path", type=str, required=True,
                        help="Path to the language model")
    parser.add_argument("--adapter-path", type=str, required=True,
                        help="Path to the CT-CHAT adapter")
    parser.add_argument("--image-file", type=str, required=True,
                        help=".npz file with the CT embeddings")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug info")
    args = parser.parse_args()

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)
        
        torch.cuda.empty_cache()
        base_model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        adapter_config_path = os.path.join(args.adapter_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            pass
        else:
            print(f"Warning: No adapter config found at {adapter_config_path}")
            
        model = PeftModel.from_pretrained(
            base_model,
            args.adapter_path,
            is_trainable=False,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        ct_data = np.load(args.image_file)
        embeddings = ct_data["arr"]
        
        batch_size = embeddings.shape[0]
        embed_dim = embeddings.shape[-1]
        spatial_dims = embeddings.shape[1:-1]
        
        flat_spatial_dim = np.prod(spatial_dims)
        embeddings_flat = embeddings.reshape(batch_size, flat_spatial_dim, embed_dim)
        embeddings_flat = torch.tensor(embeddings_flat, dtype=torch.float16).to(args.device)
        
        system_prompt = "You are an expert radiologist. Generate a comprehensive radiology report based on the provided 3D CT scan."
        user_prompt = "Analyze this chest CT scan and provide a structured report with Findings and Impression sections."
        
        prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        
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
                traceback.print_exc()
                
                try:
                    model_outputs = model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        return_dict=True,
                    )
                except Exception as e:
                    traceback.print_exc()
        
        with torch.inference_mode():
            try:
                output_ids = model.generate(
                    input_ids=inputs.input_ids,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.9,
                    max_new_tokens=800,
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
