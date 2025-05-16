#!/usr/bin/env python3
"""
Standalone script to run SPRING AMR parser
Usage: python spring_runner.py input.txt output.amr model_path
"""
import sys
import os
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_spring_parser(input_file, output_file, model_path, spring_dir):
    """
    Run SPRING parser using our simplified script
    """
    logger.info(f"Running SPRING parser on {input_file}")
    
    # Ensure spring_dir is absolute
    spring_dir = os.path.abspath(spring_dir)
    
    # Get path to our custom script
    custom_script = os.path.join(spring_dir, "bin", "parse_single_file.py")
    
    # Create the script if it doesn't exist
    if not os.path.exists(custom_script):
        logger.info(f"Creating custom parse script at {custom_script}")
        script_content = """#!/usr/bin/env python3
\"\"\"
Simplified version of SPRING's predict_amrs.py for parsing a single file.
\"\"\"
import os
import sys
import argparse
import torch
from pathlib import Path
import penman

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from spring_amr.tokenization_bart import PENMANBartTokenizer
    from spring_amr.modeling_bart import AMRBartForConditionalGeneration
    from spring_amr.utils import instantiate_model_and_tokenizer
except ImportError:
    print("Unable to import SPRING modules. Make sure SPRING is correctly installed.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input text file')
    parser.add_argument('--output', type=str, required=True, help='Output AMR file')
    parser.add_argument('--penman-linearization', action='store_true', help='Use penman linearization')
    parser.add_argument('--use-pointer-tokens', action='store_true', help='Use pointer tokens')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use (cuda or cpu)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Print arguments
    print(f"Loading model from {args.checkpoint}")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Device: {args.device}")
    
    # Load input text
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)
        
    with open(args.input, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if not lines:
        print("Input file is empty")
        sys.exit(1)
    
    print(f"Loaded {len(lines)} lines from input file")
    
    try:
        # Set device
        device = torch.device(args.device)
        print(f"Using device: {device}")
        
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        model, tokenizer = instantiate_model_and_tokenizer(
            checkpoint=args.checkpoint,
            penman_linearization=args.penman_linearization,
            use_pointer_tokens=args.use_pointer_tokens
        )
        model.to(device)
        model.eval()
        print("Model loaded successfully")
        
        # Process each line
        outputs = []
        for i, line in enumerate(lines):
            print(f"Processing line {i+1}/{len(lines)}: {line[:50]}...")
            
            # Tokenize
            x = tokenizer.batch_encode_sentences([line], device=device)
            
            # Generate
            with torch.no_grad():
                model.amr_mode = True
                out = model.generate(
                    **x,
                    max_length=512,
                    decoder_start_token_id=0,
                    num_beams=5
                )

            # Decode
            for j, tokens in enumerate(out):
                tokens = tokens.tolist()
                graph, status, (nodes, backreferences) = tokenizer.decode_amr(
                    tokens, 
                    restore_name_ops=True
                )
                amr_penman = penman.encode(graph)
                outputs.append(amr_penman)
                print(f"Successfully parsed line {i+1}")
        
        # Write outputs
        with open(args.output, 'w', encoding='utf-8') as f:
            for output in outputs:
                f.write(output + '\\n\\n')
        
        print(f"Successfully wrote {len(outputs)} AMRs to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
        with open(custom_script, 'w') as f:
            f.write(script_content)
        os.chmod(custom_script, 0o755)
        logger.info(f"Created custom parse script")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return False
    
    try:
        # Create a custom environment with PYTHONPATH set
        env = os.environ.copy()
        # Set the PYTHONPATH to include the spring_amr directory
        env['PYTHONPATH'] = f"{spring_dir}:{env.get('PYTHONPATH', '')}"
        
        # Run our custom SPRING parser command
        cmd = [
            "python", custom_script,
            "--checkpoint", model_path,
            "--input", input_file,
            "--output", output_file,
            "--penman-linearization",
            "--use-pointer-tokens"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info(f"With PYTHONPATH set to: {env['PYTHONPATH']}")
        subprocess.run(cmd, check=True, env=env)
        
        # Check if output file was created
        if not os.path.exists(output_file):
            logger.error(f"Output file was not created: {output_file}")
            return False
            
        logger.info(f"AMR parsing successful, output written to {output_file}")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running SPRING parser: {e}")
        return False
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False
