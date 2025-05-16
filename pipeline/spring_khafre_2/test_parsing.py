#!/usr/bin/env python
import sys
import torch
import penman
from pathlib import Path

# Add the current directory to the path so we can import spring_amr modules
import os
sys.path.insert(0, os.path.abspath('.'))

try:
    # Import necessary modules from spring_amr
    from spring_amr.utils import instantiate_model_and_tokenizer
    from spring_amr.tokenization_bart import PENMANBartTokenizer
    
    def parse_sentence(sentence, model_path):
        """Parse a single sentence into AMR."""
        print(f"Loading model from {model_path}")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Instantiate model and tokenizer
        print("Instantiating model and tokenizer...")
        model, tokenizer = instantiate_model_and_tokenizer(
            checkpoint=model_path,
            penman_linearization=True,
            use_pointer_tokens=True
        )
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
        
        # Tokenize the sentence
        print("Tokenizing input sentence...")
        x = tokenizer.batch_encode_sentences([sentence], device=device)
        if isinstance(x, tuple):
            if len(x) == 2:
              x, _ = x
            elif len(x) == 3:
              x, _, _ = x
            else:
            # Handle other cases if needed
              x = x[0]
        
        # Generate AMR graph tokens
        print("Generating AMR graph...")
        with torch.no_grad():
            model.amr_mode = True
            out = model.generate(
                **x,
                max_length=512,
                decoder_start_token_id=0,
                num_beams=5
            )
        
        # Decode AMR tokens to graph
        print("Decoding AMR tokens to graph...")
        tokens = out[0].tolist()
        graph, status, (nodes, backreferences) = tokenizer.decode_amr(tokens, restore_name_ops=True)
        
        return penman.encode(graph)

    def main():
        if len(sys.argv) < 2:
            print("Usage: python test_parsing.py <sentence>")
            sys.exit(1)
        
        sentence = " ".join(sys.argv[1:])
        model_path = "AMR2.parsing.pt"  # Use the .pt file directly
        
        try:
            amr_graph = parse_sentence(sentence, model_path)
            print("\nAMR Graph:")
            print(amr_graph)
        except Exception as e:
            print(f"Error during parsing: {e}")
            import traceback
            traceback.print_exc()
            
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("\nYou need to install the required dependencies in your environment:")
    print("pip install penman==1.1.0 networkx cached_property transformers==2.11.0 torch")
    print("\nIf you continue to have issues, create a new environment with Python 3.8:")
    print("conda create -n spring_env python=3.8")
    print("conda activate spring_env")
    print("pip install -r requirements.txt")
