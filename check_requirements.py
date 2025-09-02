import os
from huggingface_hub import HfApi, HfFolder

def check_huggingface_credentials():
    """Check if Hugging Face credentials are configured"""
    print_info("Checking Hugging Face credentials...")
    
    # Use the token from the environment variable
    token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print_error("HUGGING_FACE_HUB_TOKEN is not set.")
        return False
    
    # Validate the token
    try:
        api = HfApi()
        user_info = api.whoami(token=token)
        print_success(f"Logged in to Hugging Face as: {user_info['name']}")
        return True
    except Exception as e:
        print_error(f"Failed to validate Hugging Face token: {e}")
        return False

def run_basic_inference_test():
    """Run a basic inference test to verify the setup"""
    print_info("Running a basic inference test...")
    
    # Check if we have PyTorch
    if not importlib.util.find_spec("torch"):
        print_warning("PyTorch is not installed. Skipping inference test.")
        return False
    
    import torch
    
    try:
        # Run a basic CPU test
        print_info("Testing PyTorch CPU setup...")
        x = torch.ones(5, 3)
        y = x + 2
        print_success(f"Tensor operation successful: {y}")
        
        # Test a small model on CPU
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        inputs = tokenizer("Hello world!", return_tensors="pt")
        outputs = model(**inputs)
        print_success("Successfully ran a small Transformers model on CPU.")
        
        return True
    except Exception as e:
        print_error(f"Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

