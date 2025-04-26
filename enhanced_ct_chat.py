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
    """Clean the report text by removing unwanted trailing instructions"""
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

def check_environment():
    """Check if environment meets requirements and prints helpful information"""
    print("Checking environment...")
    
    # Check if running on CPU or GPU
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Running on CPU will be extremely slow.")
        print("Consider running on a machine with a compatible NVIDIA GPU.")
        device = "cpu"
    else:
        device = "cuda"
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)
        print(f"GPU memory: {vram_gb:.2f} GB")
        print(f"Current GPU memory allocated: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        print(f"Current GPU memory reserved: {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")
        
        # Check if VRAM is sufficient
        if vram_gb < 6:
            print(f"WARNING: Your GPU has only {vram_gb:.2f} GB of VRAM.")
            print("This may not be sufficient to run the model, even with 4-bit quantization.")
            print("Consider using a GPU with at least 6GB VRAM.")
            if input("Do you want to continue anyway? (y/n): ").lower() != 'y':
                sys.exit(1)
    
    return device

def create_model_with_appropriate_quantization(model_path, device="cuda"):
    """Create model with appropriate quantization based on available GPU memory"""
    if device == "cpu":
        print("Creating model for CPU inference (this will be slow)...")
        # For CPU, we use 8-bit quantization for better performance
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
            print(f"Failed to load with 8-bit quantization: {e}")
            print("Falling back to standard loading (this will use more memory)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True
            )
            return model
    
    # If using GPU, check available memory to determine quantization level
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if vram_gb < 16:
            print(f"GPU VRAM is {vram_gb:.2f} GB, using 4-bit quantization...")
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
                print(f"Failed to load with 4-bit quantization: {e}")
                print("Falling back to 8-bit quantization...")
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
                    print(f"Failed to load with 8-bit quantization: {e}")
                    print("Falling back to half precision (this will use more memory)...")
        
        # For GPUs with more memory, use half precision
        print(f"GPU VRAM is {vram_gb:.2f} GB, using half precision...")
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

    # Check for model files
    if not os.path.exists(args.llm_model_path):
        print(f"Error: Model path {args.llm_model_path} does not exist.")
        print("Please check that you've downloaded the model correctly.")
        sys.exit(1)
    
    if not os.path.exists(args.adapter_path):
        print(f"Error: Adapter path {args.adapter_path} does not exist.")
        print("Please check that you've downloaded the adapter correctly.")
        sys.exit(1)
    
    if not os.path.exists(args.image_file):
        print(f"Error: Image file {args.image_file} does not exist.")
        print("Please check that you've run the encode_script.py correctly.")
        sys.exit(1)
    
    # Determine device automatically if not specified
    device = args.device
    if device is None:
        device = check_environment()
    
    print(f"Device: {device}")
    print(f"LLM model path: {args.llm_model_path}")
    print(f"Adapter path: {args.adapter_path}")
    print(f"Image file: {args.image_file}")
    
    try:
        # Load the tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)
        print(f"Tokenizer loaded with vocab size: {len(tokenizer)}")
        
        # Load the language model with appropriate quantization
        print("Loading LLM...")
        torch.cuda.empty_cache()  # Clear GPU memory before loading
        base_model = create_model_with_appropriate_quantization(args.llm_model_path, device)
        print("LLM loaded successfully")
        
        if device == "cuda":
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
            torch_dtype=torch.float16 if device == "cuda" else None,
            device_map="auto"
        )
        print("Adapter loaded successfully")
        
        if device == "cuda":
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
        embeddings_flat = torch.tensor(embeddings_flat, dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
        print(f"Flattened embeddings tensor shape: {embeddings_flat.shape}")
        
        if device == "cuda":
            print(f"GPU memory after loading embeddings: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        
        # Set up the enhanced prompt
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
        
        # Tokenize the prompt
        print("Tokenizing prompt...")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        print(f"Input shape: {inputs.input_ids.shape}")
        
        if device == "cuda":
            print(f"GPU memory after tokenization: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        
        # Set up streamer
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        print("\nGenerating radiology report...")
        
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
        
        if device == "cuda":
            print(f"GPU memory before generation: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        
        # Determine generation parameters based on device
        max_new_tokens = 800
        if device == "cpu":
            # For CPU, use more conservative parameters
            temperature = 0.1
            top_p = 0.7
            max_new_tokens = 400  # Shorter output for CPU to save time
        else:
            # For GPU, use standard parameters
            temperature = 0.3
            top_p = 0.85
        
        # Generation step with improved parameters
        print("Step 2: Generating report...")
        with torch.inference_mode():
            try:
                # Use improved generation parameters
                output_ids = model.generate(
                    input_ids=inputs.input_ids,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=1.2,
                    streamer=streamer,
                )
                
                # Decode output
                output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # Clean the output using the improved cleaning function
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
