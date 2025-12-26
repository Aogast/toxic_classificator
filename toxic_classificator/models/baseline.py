"""
Baseline model using zero-shot Qwen without fine-tuning
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
from pathlib import Path
import yaml


class BaselineModel:
    """Baseline model for toxic comment classification using zero-shot prompting"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = None):
        """
        Initialize the baseline model
        
        Args:
            model_name: Name of the model to use
            device: Device to run the model on (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading baseline model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def create_prompt(self, text: str) -> str:
        """
        Create a prompt for classification
        
        Args:
            text: Input text to classify
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Проанализируй следующее сообщение и определи, является ли оно токсичным.

Сообщение: {text}

Ответь в формате JSON:
{{
    "toxic": true/false,
    "labels": ["список типов токсичности, если есть"]
}}

Возможные типы токсичности: severe_toxic (очень токсично), obscene (непристойно), threat (угроза), insult (оскорбление), identity_hate (ненависть к группе).

Ответ:"""
        
        return prompt
    
    def parse_response(self, response: str) -> Dict:
        """
        Parse model response to extract classification
        
        Args:
            response: Raw model response
            
        Returns:
            Dictionary with 'toxic' and 'labels' keys
        """
        try:
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                
                # Ensure required keys exist
                if 'toxic' not in result:
                    result['toxic'] = False
                if 'labels' not in result:
                    result['labels'] = []
                
                return result
            else:
                # Fallback: simple heuristic
                response_lower = response.lower()
                toxic = 'true' in response_lower or 'токсичн' in response_lower
                return {'toxic': toxic, 'labels': []}
                
        except json.JSONDecodeError:
            # Fallback parsing
            response_lower = response.lower()
            toxic = 'true' in response_lower or 'токсичн' in response_lower
            return {'toxic': toxic, 'labels': []}
    
    @torch.no_grad()
    def predict(self, text: str, max_new_tokens: int = 256) -> Dict:
        """
        Predict toxicity for a given text
        
        Args:
            text: Input text to classify
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary with 'toxic' (bool) and 'labels' (list) keys
        """
        # Create prompt
        prompt = self.create_prompt(text)
        
        # Tokenize
        messages = [{"role": "user", "content": prompt}]
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            text_input,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse response
        result = self.parse_response(response)
        
        return result
    
    def predict_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict]:
        """
        Predict toxicity for a batch of texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of dictionaries with predictions
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                result = self.predict(text)
                results.append(result)
        
        return results


def main():
    """Test the baseline model"""
    # Load config
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = BaselineModel(model_name=config['model']['name'])
    
    # Test examples
    test_texts = [
        "Привет! Как дела?",
        "Ты идиот и дурак!",
        "Я тебя ненавижу, пошел вон!",
        "Спасибо за помощь, очень полезная информация."
    ]
    
    print("\nTesting baseline model:")
    print("=" * 80)
    
    for text in test_texts:
        result = model.predict(text)
        print(f"\nText: {text}")
        print(f"Toxic: {result['toxic']}")
        print(f"Labels: {result['labels']}")
        print("-" * 80)


if __name__ == "__main__":
    main()

