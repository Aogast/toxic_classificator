"""
Triton Inference Server client for toxic classification
"""
import json
import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer
from pathlib import Path


class ToxicClassifierTritonClient:
    """Client for Triton Inference Server"""
    
    def __init__(self, triton_url: str = "localhost:8000", model_name: str = "toxic_classificator"):
        self.triton_url = triton_url
        self.model_name = model_name
        self.client = httpclient.InferenceServerClient(url=triton_url)
        
        # Load tokenizer
        tokenizer_path = "triton_model_repository/toxic_classificator/1/merged_model"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        print(f"Connected to Triton at {triton_url}")
        print(f"Model: {model_name}")
    
    def predict(self, text: str, max_length: int = 128) -> dict:
        """
        Predict toxicity for a single text
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with prediction results
        """
        # Prepare input
        prompt = f"Human: {text}\nAssistant:"
        inputs = self.tokenizer(
            prompt,
            return_tensors="np",
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)
        
        # Create Triton inputs
        triton_inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
            httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
        ]
        
        triton_inputs[0].set_data_from_numpy(input_ids)
        triton_inputs[1].set_data_from_numpy(attention_mask)
        
        # Create Triton outputs
        triton_outputs = [
            httpclient.InferRequestedOutput("logits")
        ]
        
        # Send request
        response = self.client.infer(
            model_name=self.model_name,
            inputs=triton_inputs,
            outputs=triton_outputs
        )
        
        # Get logits
        logits = response.as_numpy("logits")
        
        # Generate tokens (simple greedy decoding)
        # For production, you'd use beam search or sampling
        generated_ids = np.argmax(logits[0], axis=-1)
        
        # Decode response
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Parse response
        toxic = "toxic" in response_text.lower()
        
        return {
            "text": text,
            "response": response_text,
            "toxic": toxic,
            "labels": []
        }
    
    def predict_batch(self, texts: list, max_length: int = 128) -> list:
        """
        Predict toxicity for multiple texts
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for text in texts:
            result = self.predict(text, max_length)
            results.append(result)
        return results


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Triton client for toxic classification")
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument("--input_file", type=str, help="Input file with texts")
    parser.add_argument("--output_file", type=str, help="Output file for results")
    parser.add_argument("--triton_url", type=str, default="localhost:8000", help="Triton server URL")
    
    args = parser.parse_args()
    
    # Create client
    client = ToxicClassifierTritonClient(triton_url=args.triton_url)
    
    # Get texts
    if args.text:
        texts = [args.text]
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            if args.input_file.endswith(".json"):
                data = json.load(f)
                texts = [item["text"] for item in data]
            else:
                texts = [line.strip() for line in f if line.strip()]
    else:
        print("Error: Provide --text or --input_file")
        return
    
    # Predict
    print(f"\nProcessing {len(texts)} texts...")
    results = client.predict_batch(texts)
    
    # Output
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {args.output_file}")
    else:
        for result in results:
            print(f"\nText: {result['text']}")
            print(f"Response: {result['response']}")
            print(f"Toxic: {result['toxic']}")


if __name__ == "__main__":
    main()

