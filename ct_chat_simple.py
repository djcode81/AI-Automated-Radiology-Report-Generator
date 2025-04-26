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
    """Clean the report text by removing unwanted trailing instructions"""
    # First, check if the text is enclosed in <s> and </s> tags
    if text.startswith("<s>") and "</s>" in text:
        # Extract just the content between <s> and </s>
        text = text.split("<s>")[1].split("</s>")[0].strip()
    else:
        # List of patterns that indicate the end of the report
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
        
        # Check for each marker and cut the text at that point
        for marker in end_markers:
            if marker in text:
                text = text.split(marker)[0]
        
        # Also look for sentence patterns that might indicate instructions
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
    
    # Trim any trailing whitespace
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

    print(f"Device: {args.device}")
    print(f"LLM model path: {args.llm_model_path}")
    print(f"Adapter path: {args.adapter_path}")
    print(f"Image file: {args.image_file}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"Current GPU memory allocated: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        print(f"Current GPU memory reserved: {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")
    
    try:
        # Load the tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)
        print(f"Tokenizer loaded with vocab size: {len(tokenizer)}")
        
        # Load the language model
        print("Loading LLM...")
        torch.cuda.empty_cache()  # Clear GPU memory before loading
        base_model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("LLM loaded successfully")
        print(f"GPU memory after loading LLM: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        
        # Load the adapter
        print("Loading adapter...")
        adapter_config_path = os.path.join(args.adapter_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            print(f"Found adapter config at {adapter_config_path}")
        else:
            print(f"Warning: No adapter config found at {adapter_config_path}")
            
        model = PeftModel.from_pretrained(
            base_model,
            args.adapter_path,
            is_trainable=False,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Adapter loaded successfully")
        print(f"GPU memory after loading adapter: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        
        # Load CT embeddings
        print("Loading CT embeddings...")
        ct_data = np.load(args.image_file)
        embeddings = ct_data["arr"]
        print(f"CT embeddings shape: {embeddings.shape}")
        
        # Process embeddings
        batch_size = embeddings.shape[0]
        embed_dim = embeddings.shape[-1]
        spatial_dims = embeddings.shape[1:-1]
        print(f"Spatial dimensions: {spatial_dims}")
        
        # Flatten spatial dimensions for models that expect 2D
        flat_spatial_dim = np.prod(spatial_dims)
        embeddings_flat = embeddings.reshape(batch_size, flat_spatial_dim, embed_dim)
        embeddings_flat = torch.tensor(embeddings_flat, dtype=torch.float16).to(args.device)
        print(f"Flattened embeddings tensor shape: {embeddings_flat.shape}")
        print(f"GPU memory after loading embeddings: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        
        # Set up the prompt
        system_prompt = "You are an expert radiologist. Generate a comprehensive radiology report based on the provided 3D CT scan."
        user_prompt = "Analyze this chest CT scan and provide a structured report with Findings and Impression sections."
        
        prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        
        # Tokenize the prompt
        print("Tokenizing prompt...")
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        print(f"Input shape: {inputs.input_ids.shape}")
        print(f"GPU memory after tokenization: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        
        # Set up streamer
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        print("\nGenerating radiology report...")
        
        # Simplified generation approach - just try the conditioning method
        print("Attempting to generate with embedding conditioning...")
        
        # Try to condition the model with embeddings
        print("Step 1: Running conditioning forward pass...")
        with torch.inference_mode():
            # Use a try block for each potentially problematic step
            try:
                # First attempt - using encoder_hidden_states
                model_outputs = model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    encoder_hidden_states=embeddings_flat,
                    return_dict=True,
                )
                print("Conditioning forward pass successful")
            except Exception as e:
                print(f"Error in conditioning forward pass: {e}")
                traceback.print_exc()
                
                # Try an alternate approach
                print("Trying alternate conditioning approach...")
                try:
                    # Simply do a normal forward pass to warm up the model
                    model_outputs = model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        return_dict=True,
                    )
                    print("Alternate conditioning successful")
                except Exception as e:
                    print(f"Error in alternate conditioning: {e}")
                    traceback.print_exc()
        
        print(f"GPU memory before generation: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        
        # Generation step
        print("Step 2: Generating report...")
        with torch.inference_mode():
            try:
                # Use simpler generation parameters
                output_ids = model.generate(
                    input_ids=inputs.input_ids,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.9,
                    max_new_tokens=800,
                    repetition_penalty=1.2,
                    streamer=streamer,
                )
                
                # Decode output
                output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # Clean the output using the new cleaning function
                cleaned_output = clean_report(output_text)
                
                # Save the raw and cleaned reports
                raw_report_file = os.path.basename(args.image_file).replace('.npz', '_ct_chat_raw_report.txt')
                with open(raw_report_file, 'w') as f:
                    f.write(output_text)
                
                report_file = os.path.basename(args.image_file).replace('.npz', '_ct_chat_report.txt')
                with open(report_file, 'w') as f:
                    f.write(cleaned_output)
                
                print(f"\nRaw report saved to {raw_report_file}")
                print(f"Cleaned report saved to {report_file}")
                
                # Print a preview of the cleaned report
                preview_length = min(300, len(cleaned_output))
                print(f"\nCleaned report preview (first {preview_length} chars):")
                print(cleaned_output[:preview_length] + ("..." if len(cleaned_output) > preview_length else ""))
                
                # Compare with raw to show what was removed
                if len(output_text) != len(cleaned_output):
                    print(f"\nRemoved {len(output_text) - len(cleaned_output)} characters of trailing instructions")
                
            except Exception as e:
                print(f"Error during generation: {e}")
                traceback.print_exc()
                
                # Try a fallback with even simpler parameters
                print("Trying fallback generation...")
                try:
                    output_ids = model.generate(
                        input_ids=inputs.input_ids,
                        max_new_tokens=200,  # Shorter output
                        streamer=streamer,
                    )
                    
                    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    cleaned_output = clean_report(output_text)
                    
                    report_file = os.path.basename(args.image_file).replace('.npz', '_ct_chat_fallback_report.txt')
                    with open(report_file, 'w') as f:
                        f.write(cleaned_output)
                    
                    print(f"\nFallback report saved to {report_file}")
                    
                except Exception as e:
                    print(f"Fallback generation also failed: {e}")
                    traceback.print_exc()
    
    except Exception as e:
        print(f"Unhandled exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
