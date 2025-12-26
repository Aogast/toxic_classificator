"""
Convert merged PyTorch model to ONNX using optimum
"""
from pathlib import Path
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

def main():
    merged_model_path = "triton_model_repository/toxic_classificator/1/merged_model"
    onnx_output_path = "triton_model_repository/toxic_classificator/1/onnx_model"
    
    print(f"Loading merged model from: {merged_model_path}")
    
    # Load and convert to ONNX
    print("Converting to ONNX...")
    model = ORTModelForCausalLM.from_pretrained(
        merged_model_path,
        export=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
    
    # Save ONNX model
    print(f"Saving ONNX model to: {onnx_output_path}")
    model.save_pretrained(onnx_output_path)
    tokenizer.save_pretrained(onnx_output_path)
    
    print(f"✅ ONNX model saved to: {onnx_output_path}")

if __name__ == "__main__":
    main()

