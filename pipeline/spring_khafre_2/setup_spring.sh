#!/bin/bash
# Setup script for SPRING AMR parser
set -e

echo "Setting up SPRING AMR parser environment..."

# Create directories
mkdir -p spring_amr
mkdir -p pipeline_out

# Clone SPRING repository if not already done
if [ ! -d "spring_amr/.git" ]; then
    echo "Cloning SPRING repository..."
    rm -rf spring_amr/*
    git clone https://github.com/SapienzaNLP/spring.git spring_amr_temp
    mv spring_amr_temp/* spring_amr/
    mv spring_amr_temp/.git spring_amr/
    rm -rf spring_amr_temp
    echo "SPRING repository cloned successfully"
else
    echo "SPRING repository already exists, skipping clone"
fi

# Download model if needed
if [ ! -f "AMR2.parsing.pt" ]; then
    echo "Downloading SPRING AMR model..."
    wget -O AMR2.parsing.pt https://github.com/SapienzaNLP/spring/releases/download/0.1.0/AMR2.parsing.pt
    echo "Model downloaded successfully"
else
    echo "Model already exists, skipping download"
fi

# Create a special environment for SPRING
if ! conda env list | grep -q "spring_env"; then
    echo "Creating conda environment for SPRING..."
    conda create -y -n spring_env python=3.8
    conda activate spring_env
    
    # Install dependencies for SPRING
    echo "Installing SPRING dependencies..."
    pip install torch==1.7.1
    pip install transformers==4.2.2
    pip install tokenizers==0.9.4
    pip install penman==1.1.0
    pip install networkx
    pip install sentencepiece
    pip install cached_property
    
    echo "Installing dependencies for main script..."
    pip install rdflib
    pip install tqdm
    
    echo "Dependencies installed successfully"
else
    echo "Spring environment already exists, skipping creation"
    conda activate spring_env
fi

# Create a simple test for SPRING
echo "Creating test files..."
echo "The cat sits on the mat." > test_input.txt

# Create the runner script
echo "Creating SPRING runner script..."
cat > spring_runner.py << 'EOF'
#!/usr/bin/env python3
"""
Standalone script to run SPRING AMR parser
Usage: python spring_runner.py input.txt output.amr spring_dir
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
    Run SPRING parser using the original SPRING scripts
    """
    logger.info(f"Running SPRING parser on {input_file}")
    
    # Ensure spring_dir is absolute
    spring_dir = os.path.abspath(spring_dir)
    
    # Get paths
    predict_script = os.path.join(spring_dir, "bin", "predict_amrs.py")
    
    # Check if files exist
    if not os.path.exists(predict_script):
        logger.error(f"SPRING predict script not found at {predict_script}")
        return False
        
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return False
    
    try:
        # Run SPRING parser command
        cmd = [
            "python", predict_script,
            "--checkpoint", model_path,
            "--input", input_file,
            "--output", output_file,
            "--penman-linearization",
            "--use-pointer-tokens"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
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

def main():
    if len(sys.argv) != 4:
        print("Usage: python spring_runner.py input.txt output.amr spring_dir")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    spring_dir = sys.argv[3]
    
    # Default model path - adjust if needed
    model_path = os.path.abspath("./AMR2.parsing.pt")
    
    # Run parser
    success = run_spring_parser(input_file, output_file, model_path, spring_dir)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# Make the runner script executable
chmod +x spring_runner.py

echo "Setup complete!"
echo "To test if SPRING is working, run:"
echo "python spring_runner.py test_input.txt test_output.amr ./spring_amr"
echo ""
echo "Then run the main script with:"
echo "python khafre_spring.py --input your_graph.ttl"
