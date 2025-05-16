#!/usr/bin/env python3
"""
Simple AMR generator that creates basic AMR representations without using SPRING
"""
import sys
import os
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_simple_amr(text):
    """
    Generate a simple AMR representation from text
    """
    # Basic preprocessing
    text = text.strip()
    if not text:
        return "(a / amr-empty)"
        
    # Extract main components with regex
    # This is very simplified and won't work for complex sentences
    pattern = r'(The |A |An )?(\w+)\s+(\w+s?)\s+(.*)'
    match = re.match(pattern, text)
    
    if match:
        # Extract potential subject, verb, object
        _, subject, verb, remainder = match.groups()
        
        # Build a simple AMR representation
        amr = f"(v / {verb}-01\n"
        amr += f"    :ARG0 (s / {subject})\n"
        
        if remainder:
            amr += f"    :ARG1 (o / object\n"
            amr += f"        :mod (t / text\n"
            amr += f"            :mod \"{remainder}\")))\n"
        else:
            amr += ")\n"
            
        return amr
    else:
        # Fallback for unmatched patterns
        return f"(t / thing\n    :ARG0-of (h / have-mod-91\n        :ARG1 (m / message\n            :mod \"{text}\")))\n"

def main():
    if len(sys.argv) != 3:
        print("Usage: python simple_amr_generator.py input.txt output.amr")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    logger.info(f"Reading from {input_file}")
    logger.info(f"Writing to {output_file}")
    
    try:
        # Read input text
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            sys.exit(1)
            
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Generate AMR
        amr = generate_simple_amr(text)
        
        # Write output
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(amr)
            
        logger.info(f"Successfully wrote AMR to {output_file}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
