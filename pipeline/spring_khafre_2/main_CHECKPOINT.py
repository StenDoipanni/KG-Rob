from rdflib import Graph
import logging
import os
import json
import re
import requests
import argparse
import time
import sys
import subprocess
import io
from util import *
#from visualization import DigraphWriter

# Import the SPRING parser dependencies
import torch
import penman
from pathlib import Path
from spring_amr import postprocessing
from penman import encode, decode
from spring_amr.tokenization_bart import AMRBartTokenizer

# Download required NLTK data
import nltk
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Configure logging to capture to file
pipeline_dir = "./pipeline_out/"
os.makedirs(pipeline_dir, exist_ok=True)

# Create log file handler
log_file = os.path.join(pipeline_dir, "log.txt")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(), file_handler])
logger = logging.getLogger(__name__)

# Replace this with your test text if you want to skip Ollama
# If left empty, the script will call Ollama for verbalization
test_text = ""

# Ollama model to use
ollama_model = "llama3.1:8b"

# SPRING model file to use - updated to AMR3.parsing.pt
spring_model_file = "AMR3.parsing.pt"


def save_graph_visualizations(graph, base_filename, output_dir):
    """
    Saves both PNG and SVG visualizations of a graph
    
    Args:
        graph: The RDF graph to visualize
        base_filename: Base name for the output files (without extension)
        output_dir: Directory to save the visualizations
    """
    logger.info(f"Generating visualizations for {base_filename}")
    
    # Create a DigraphWriter instance
    digraph_writer = DigraphWriter()
    
    # Generate DOT format with empty not_visible_graph
    dot_content = digraph_writer.graph_to_digraph(graph, not_visible_graph=set())
    
    # Save DOT format for debugging
    dot_file = os.path.join(output_dir, f"{base_filename}.dot")
    with open(dot_file, "w") as f:
        f.write(dot_content)
    logger.info(f"Saved DOT format to {dot_file}")
    
    # Use subprocess to call graphviz directly for PNG
    try:
        png_file = os.path.join(output_dir, f"{base_filename}.png")
        subprocess.run(['dot', '-Tpng', dot_file, '-o', png_file], check=True)
        logger.info(f"Saved PNG visualization to {png_file}")
    except Exception as e:
        logger.error(f"Error generating PNG visualization: {e}")
    
    # Use subprocess to call graphviz directly for SVG
    try:
        svg_file = os.path.join(output_dir, f"{base_filename}.svg")
        subprocess.run(['dot', '-Tsvg', dot_file, '-o', svg_file], check=True)
        logger.info(f"Saved SVG visualization to {svg_file}")
    except Exception as e:
        logger.error(f"Error generating SVG visualization: {e}")


def extract_base_terms_from_graph(graph):
    """
    Extracts base terms from an existing graph by:
    1. Finding entities with 'schematicRelation' in their URI
    2. Getting their types as base terms
    3. Finding related objects through predicates starting with 'has'
    """
    # this has to be double checked, what if we have relations between objects of the same type?
    # e.g. what if we have "The knife is touching the knife"? 
    base_terms = set() 
    # this is ok for storing image schematic relations
    salient_relations = []
    
    # SPARQL query to find schematic relations and their types
    query1 = """
    SELECT DISTINCT ?schematicRelation ?type
    WHERE {
        ?schematicRelation a ?type .
        FILTER(CONTAINS(str(?schematicRelation), "schematicRelation"))
        FILTER(!CONTAINS(str(?type), "NamedIndividual"))
    }
    """
    
    # SPARQL query to find objects related to schematic relations through 'has' predicates
    query2 = """
    SELECT DISTINCT ?schematicRelation ?predicate ?object
    WHERE {
        ?schematicRelation ?predicate ?object .
        FILTER(CONTAINS(str(?schematicRelation), "schematicRelation"))
        FILTER(CONTAINS(str(?predicate), "has"))
    }
    """
    
    try:
        # Execute first query to get relation types
        logger.info("Extracting relation types as base terms...")
        results1 = graph.query(query1)
        
        for row in results1:
            type_uri = str(row['type'])
            relation_uri = str(row['schematicRelation'])
            
            # Extract the type name without prefix
            type_name = type_uri.split('/')[-1].split('#')[-1]
            
            if type_name and type_name not in base_terms:
                base_terms.add(type_name)
                salient_relations.append(f"{relation_uri} a {type_uri}")
                logger.info(f"Added relation type as base term: {type_name}")
        
        # Execute second query to get related objects
        logger.info("Extracting related objects as base terms...")
        results2 = graph.query(query2)
        
        for row in results2:
            object_uri = str(row['object'])
            relation_uri = str(row['schematicRelation'])
            predicate_uri = str(row['predicate'])
            
            # Extract the object name without prefix and numeric suffix
            object_name = object_uri.split('/')[-1].split('#')[-1]
            object_base = object_name.split('_')[0] if '_' in object_name else object_name
            
            if object_base and object_base not in base_terms:
                base_terms.add(object_base)
                salient_relations.append(f"{relation_uri} {predicate_uri} {object_uri}")
                logger.info(f"Added related object as base term: {object_base}")
        
    except Exception as e:
        logger.error(f"Error extracting base terms: {e}")
    
    # Save salient relations to file
    salient_file = os.path.join(pipeline_dir, "salient_relations.txt")
    with open(salient_file, "w") as f:
        f.write("\n".join(salient_relations))
    logger.info(f"Saved salient relations to {salient_file}")
    
    # Convert set to list for return
    base_terms_list = list(base_terms)
    logger.info(f"Extracted {len(base_terms_list)} base terms: {base_terms_list}")
    
    return base_terms_list


def extract_schematic_relations(graph):
    """
    Extract schematic relations from the graph as separate subgraphs
    """
    relations = []
    
    # Query to find all schematic relations
    query = """
    SELECT DISTINCT ?relation
    WHERE {
        ?relation a ?type .
        FILTER(CONTAINS(str(?relation), "schematicRelation"))
    }
    """
    
    try:
        results = graph.query(query)
        
        for row in results:
            relation_uri = row['relation']
            
            # Create a subgraph for this relation
            relation_graph = Graph()
            
            # Find all triples related to this relation
            for s, p, o in graph.triples((relation_uri, None, None)):
                relation_graph.add((s, p, o))
                
                # Also add type information for objects
                if p.n3().startswith('<http') and p.n3().find('has') >= 0:
                    for s2, p2, o2 in graph.triples((o, None, None)):
                        relation_graph.add((s2, p2, o2))
            
            # Add the relation subgraph to the list
            # This whole part is not really needed, we can just return the relation_graph
            if len(relation_graph) > 0:
                relations.append(relation_graph)
                
                # Print triples for this relation with stripped URIs
                print(f"\nTriples for relation:")
                for s, p, o in relation_graph:
                    # Strip URIs for display using the same logic as _extract_base_name
                    s_str = str(s)
                    p_str = str(p)
                    o_str = str(o)
                    
                    # Strip subject URI
                    if '://' in s_str:
                        s_base = s_str.split('/')[-1].split('#')[-1]
                        s_base = s_base.split('_')[0] if '_' in s_base else s_base
                        s_base = s_base.lower()
                    else:
                        s_base = s_str
                        
                    # Strip predicate URI
                    if '://' in p_str:
                        p_base = p_str.split('/')[-1].split('#')[-1]
                        p_base = p_base.split('_')[0] if '_' in p_base else p_base
                        p_base = p_base.lower()
                    else:
                        p_base = p_str
                        
                    # Strip object URI
                    if '://' in o_str:
                        o_base = o_str.split('/')[-1].split('#')[-1]
                        o_base = o_base.split('_')[0] if '_' in o_base else o_base
                        o_base = o_base.lower()
                    else:
                        o_base = o_str
                        
                    print(f"  {s_base} {p_base} {o_base}")
    
    except Exception as e:
        logger.error(f"Error extracting schematic relations: {e}")
    
    return relations


def get_verbalization_from_ollama(triples, model, output_dir="./out/ollama/"):
    """
    Get verbalization from Ollama for the given triples and log both prompt and response
    """
    logger.info(f"=========== CALLING OLLAMA API ===========")
    logger.info(f"Using model: {model}")
    
    prompt = f"""
You receive some triples in Turtle language, and you have to provide a simple verbalization.
Here is an example.
Example Triples:
ns2:schematicRelation_14 a ns2:Occlusion,
        owl:NamedIndividual ;
    ns2:eventMode ns2:Ended ;
    ns2:hasOccludee ns2:knife_70 ;
    ns2:hasOccluder ns2:apple_27 .

"Verbalization": The knife is occluding the apple.

Answer according to the template:
"Verbalization": your answer here

and nothing more, disregard numbers, the verbalization should be related only to the provided triples, and not the ones in example.
Do not overthink, it is a simple task.


Triples:
{triples}

"Verbalization":
"""
    
    # Create a unique identifier for this call
    call_id = f"ollama_call_{int(time.time())}"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Log the prompt to a file
    prompt_file = os.path.join(output_dir, f"{call_id}_prompt.txt")
    with open(prompt_file, "w") as f:
        f.write(prompt)
    
    logger.info(f"Prompt saved to: {prompt_file}")
    logger.info(f"Sending prompt to Ollama:")
    # Log a preview of the triples for debugging, truncated to avoid cluttering logs
    logger.info(f"First 200 chars of triples in prompt: {triples[:200]}...")

    try:
        # Make request to Ollama API
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True  # Stream for live output
        }
        
        logger.info(f"Sending request to Ollama API endpoint")
        
        print("\n=========== OLLAMA LIVE OUTPUT ===========")
        # For streaming we need to use a different approach
        response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                # Parse the JSON from each line
                try:
                    json_line = json.loads(line.decode('utf-8'))
                    if 'response' in json_line:
                        chunk = json_line['response']
                        full_response += chunk
                        print(chunk, end='', flush=True)
                except json.JSONDecodeError:
                    pass
        
        print("\n=========================================")
        
        # Save the full response to a file
        ollama_output_file = os.path.join(pipeline_dir, "ollama_output.txt")
        with open(ollama_output_file, "w") as f:
            f.write(full_response)
            
        logger.info(f"Saved full Ollama response to: {ollama_output_file}")
        
        # Try to extract the verbalization using regex
        verbalization = ""
        match = re.search(r'"Verbalization":\s*"([^"]+)"', full_response)
        if match:
            verbalization = match.group(1)
            logger.info(f"Extracted verbalization using regex: '{verbalization}'")
        else:
            # Try to find any text between quotes
            match = re.search(r'"([^"]+)"', full_response)
            if match:
                verbalization = match.group(1)
                logger.info(f"Extracted quoted text: '{verbalization}'")
            else:
                # Use the entire response as fallback
                verbalization = full_response.strip()
                logger.info(f"Using raw response as verbalization: '{verbalization}'")
        
        return verbalization
            
    except Exception as e:
        error_msg = f"Exception when calling Ollama: {e}"
        logger.error(error_msg)
        return "The object is related to another object."


def get_verbalization_from_anthropic(triples, model="claude-3-7-sonnet-20250219", api_key=None, output_dir="./out/anthropic/"):
    """
    Get verbalization from Anthropic's Claude API for the given triples
    """
    if not api_key:
        logger.error("No API key provided for Anthropic")
        return "The object is related to another object."
        
    logger.info(f"=========== CALLING ANTHROPIC API ===========")
    logger.info(f"Using model: {model}")
    
    prompt = f"""
You receive some triples in Turtle language, and you have to provide a simple verbalization.
Here is an example.
Example Triples:
ns2:schematicRelation_14 a ns2:Occlusion,
        owl:NamedIndividual ;
    ns2:eventMode ns2:Ended ;
    ns2:hasOccludee ns2:knife_70 ;
    ns2:hasOccluder ns2:apple_27 .

"Verbalization": The knife is occluding the apple.

Answer according to the template:
"Verbalization": your answer here

and nothing more, disregard numbers, the verbalization should be related only to the provided triples, and not the ones in example.
Do not overthink, it is a simple task.


Triples:
{triples}

"Verbalization":
"""
    
    # Create a unique identifier for this call
    call_id = f"anthropic_call_{int(time.time())}"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Log the prompt to a file
    prompt_file = os.path.join(output_dir, f"{call_id}_prompt.txt")
    with open(prompt_file, "w") as f:
        f.write(prompt)
    
    logger.info(f"Prompt saved to: {prompt_file}")
    logger.info(f"Sending prompt to Anthropic API:")
    logger.info(f"First 200 chars of triples in prompt: {triples[:200]}...")

    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=api_key)
        
        message = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0,
            system="You are a helpful assistant that verbalizes RDF triples into natural language.",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        full_response = message.content[0].text
        
        # Save the full response to a file
        anthropic_output_file = os.path.join(pipeline_dir, "anthropic_output.txt")
        with open(anthropic_output_file, "w") as f:
            f.write(full_response)
            
        logger.info(f"Saved full Anthropic response to: {anthropic_output_file}")
        
        # Try to extract the verbalization using regex
        verbalization = ""
        match = re.search(r'"Verbalization":\s*"([^"]+)"', full_response)
        if match:
            verbalization = match.group(1)
            logger.info(f"Extracted verbalization using regex: '{verbalization}'")
        else:
            # Try to find any text between quotes
            match = re.search(r'"([^"]+)"', full_response)
            if match:
                verbalization = match.group(1)
                logger.info(f"Extracted quoted text: '{verbalization}'")
            else:
                # Use the entire response as fallback
                verbalization = full_response.strip()
                logger.info(f"Using raw response as verbalization: '{verbalization}'")
        
        return verbalization
            
    except Exception as e:
        error_msg = f"Exception when calling Anthropic API: {e}"
        logger.error(error_msg)
        return "The object is related to another object."


def serialize_graph_as_turtle(graph):
    """
    Serialize a graph as Turtle format
    """
    return graph.serialize(format="turtle")


def load_spring_model(model_file=spring_model_file):
    """
    Load the SPRING AMR parser model
    """
    try:
        logger.info(f"Loading SPRING model: {model_file}")
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to get to the implementation directory
        implementation_dir = os.path.dirname(script_dir)
        # Get the spring_amr directory
        spring_amr_path = os.path.join(implementation_dir, "spring", "spring_amr")
        
        # Check if setup.py exists and install the package if needed
        setup_py = os.path.join(spring_amr_path, "setup.py")
        if os.path.exists(setup_py):
            logger.info("Installing SPRING package...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-e", spring_amr_path], check=True)
                logger.info("SPRING package installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install SPRING package: {e}")
                return None
        
        # Add the spring_amr directory to Python path
        if spring_amr_path not in sys.path:
            sys.path.insert(0, spring_amr_path)
        
        # Import SPRING-specific modules here to avoid issues if they're not available
        try:
            from spring_amr.penman import encode
            from spring_amr.utils import instantiate_model_and_tokenizer
        except ImportError as e:
            logger.error(f"Failed to import SPRING modules: {e}")
            logger.error(f"Python path: {sys.path}")
            logger.error(f"Looking for spring_amr in: {spring_amr_path}")
            return None
        
        # Get the root directory of SPRING
        spring_dir = Path(spring_amr_path)
        
        if not spring_dir.exists():
            logger.error(f"SPRING directory not found at {spring_dir}")
            return None
            
        # Check if model file exists
        model_path = spring_dir / model_file
        if not model_path.exists():
            logger.error(f"Model file not found at {model_path}")
            return None
            
        # Load the model and tokenizer
        logger.info(f"Instantiating model from {model_path}")
        
        # First load the checkpoint to get the vocabulary size
        checkpoint = torch.load(model_path, map_location='cpu')
        vocab_size = checkpoint['model']['model.shared.weight'].shape[0]
        logger.info(f"Using vocabulary size from checkpoint: {vocab_size}")
        
        # Initialize model with BART configuration
        model, tokenizer = instantiate_model_and_tokenizer(
            name='facebook/bart-large',
            checkpoint=None,  # Don't load weights yet
            additional_tokens_smart_init=True,
            dropout=0.15,
            attention_dropout=0.15,
            from_pretrained=True,
            init_reverse=False,
            collapse_name_ops=True,
            penman_linearization=False,
            use_pointer_tokens=True,  # Keep pointer tokens
            raw_graph=False
        )
        
        # Resize token embeddings to match checkpoint
        model.resize_token_embeddings(vocab_size)
        
        # Now load the state dict
        model.load_state_dict(checkpoint['model'])
        
        # Move model to GPU if available, otherwise CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        logger.info(f"SPRING model loaded successfully on {device}")
        
        return (model, tokenizer, device)
    except Exception as e:
        logger.error(f"Error loading SPRING model: {e}")
        logger.exception("Detailed error information:")
        return None


def convert_pointers_to_z_format(amr_string):
    """
    Convert AMR with pointer tokens to traditional z variable format
    """
    # Remove duplicate pointer:0 at the start
    amr_string = re.sub(r'^<pointer:0>\s*<pointer:0>', '(', amr_string)
    
    # Convert pointer tokens to z variables
    def replace_pointer(match):
        pointer_num = match.group(1)
        return f"z{pointer_num}"
    
    # Replace all pointer tokens
    amr_string = re.sub(r'<pointer:(\d+)>', replace_pointer, amr_string)
    
    # Add proper spacing and slashes
    amr_string = re.sub(r'(z\d+)\s*([a-zA-Z-]+)', r'\1 / \2', amr_string)
    
    # Clean up literals
    amr_string = re.sub(r'<lit>\s*(.*?)\s*</lit>', r'"\1"', amr_string)
    
    # Add newlines for better readability (optional)
    amr_string = re.sub(r'\)\s*:', ')\n    :', amr_string)
    amr_string = re.sub(r'\(\s*(z\d+)', '\n    (\\1', amr_string)
    
    return amr_string


def get_amr_from_spring_parser(text, spring_model=None, output_dir="./out/amr/"):
    """
    Get AMR representation using the SPRING parser
    """
    logger.info(f"=========== USING SPRING AMR PARSER ===========")
    logger.info(f"Parsing text: '{text}'")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save text to temporary file
    temp_input_file = os.path.join(output_dir, "input.txt")
    with open(temp_input_file, "w") as f:
        f.write(text)
    
    try:
        if spring_model is None:
            # Try loading the model
            spring_model = load_spring_model()
            
        if spring_model is None:
            logger.error("SPRING model could not be loaded, falling back to simple AMR generator")
            return get_amr_from_local_parser(text, output_dir)
            
        model, tokenizer, device = spring_model
        
        # Process the input text
        logger.info("Processing input with SPRING parser")
        
        # Prepare the input with proper tokenization
        input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
        
        # Generate the AMR graph
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                decoder_start_token_id=0,
                num_beams=1,
                num_return_sequences=1,
                max_length=1024
            )
            
        # Decode the outputs using the tokenizer
        decoded_amr = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Convert pointer format to z format
        traditional_amr = convert_pointers_to_z_format(decoded_amr)
        
        # Save to the pipeline directory
        output_file = os.path.join(pipeline_dir, "amr.txt")
        with open(output_file, "w") as f:
            f.write(traditional_amr)
        
        logger.info(f"AMR parsing successful. Output saved to {output_file}")
        
        # Print the AMR to terminal
        print("\n=========== AMR REPRESENTATION ===========")
        print(traditional_amr)
        print("========================================\n")
        
        return traditional_amr
        
    except Exception as e:
        error_msg = f"Error using SPRING parser: {e}"
        logger.error(error_msg)
        logger.exception("Detailed error information:")
        logger.warning("Falling back to simple AMR generator")
        
        # Fall back to the simple AMR generator
        return get_amr_from_local_parser(text, output_dir)


def get_amr_from_local_parser(text, output_dir="./out/amr/"):
    """
    TESTING PURPOSES ONLY - DO NOT USE IN PRODUCTION
    
    This function is provided for testing purposes only and should not be used as a fallback
    for the SPRING parser. It uses a simple rule-based AMR generator that produces very basic
    and potentially incorrect AMR structures.
    
    If you need to use this function for testing, uncomment the code below.
    """
    logger.error("Simple AMR generator is disabled. Please use the SPRING parser instead.")
    return None
    
    # The following code is commented out and should only be used for testing purposes
    """
    logger.info(f"=========== USING SIMPLE AMR GENERATOR ===========")
    logger.info(f"Parsing text: '{text}'")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save text to temporary file
    temp_input_file = os.path.join(output_dir, "input.txt")
    with open(temp_input_file, "w") as f:
        f.write(text)
    
    # Output file for AMR
    temp_output_file = os.path.join(output_dir, "output.amr")
    
    try:
        # Path to the simple AMR generator script
        generator_script = os.path.abspath("./simple_amr_generator.py")
        
        # Call the generator script
        cmd = ["python", generator_script, temp_input_file, temp_output_file]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Log the output
        logger.info(f"STDOUT: {result.stdout}")
        if result.stderr:
            logger.error(f"STDERR: {result.stderr}")
            
        if result.returncode != 0:
            logger.error(f"AMR generator failed with return code {result.returncode}")
            return None
        
        # Check if output file exists
        if not os.path.exists(temp_output_file):
            logger.error(f"Output file {temp_output_file} was not created")
            return None
        
        # Read the AMR output file
        with open(temp_output_file, "r") as f:
            amr_output = f.read().strip()
        
        if not amr_output:
            logger.error("Output file is empty")
            return None
            
        # Save to the pipeline directory
        output_file = os.path.join(pipeline_dir, "amr.txt")
        with open(output_file, "w") as f:
            f.write(amr_output)
        
        logger.info(f"AMR parsing successful. Output saved to {output_file}")
        
        # Print the AMR to terminal
        print("\n=========== AMR REPRESENTATION ===========")
        print(amr_output)
        print("========================================\n")
        
        return amr_output
    
    except Exception as e:
        error_msg = f"Error parsing text to AMR: {e}"
        logger.error(error_msg)
        logger.exception("Detailed error information:")
        return None
    """


def main():
    # Global declarations
    global ollama_model
    global spring_model_file

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process RDF graphs with AMR2FRED and Ollama using SPRING AMR parser")
    parser.add_argument("--input", "-i", type=str, help="Path to input graph file", required=True)
    parser.add_argument("--model", "-m", type=str, default="llama3.1:8b", help="Ollama model to use")
    parser.add_argument("--use-spring", "-s", action="store_true", help="Use SPRING parser instead of simple AMR generator")
    parser.add_argument("--spring-model", type=str, default=spring_model_file, help="SPRING model file to use")
    parser.add_argument("--online-model", type=str, help="Online model to use (e.g. claude-3-7-sonnet-20250219)")
    parser.add_argument("--api-key", type=str, help="API key for online model")
    args = parser.parse_args()
    
    # Set Ollama model
    if args.model:
        ollama_model = args.model
    
    # Set SPRING model file
    if args.spring_model:
        spring_model_file = args.spring_model
    
    # Load your existing graph
    existing_graph = Graph()
    input_graph_path = args.input
    
    try:
        existing_graph.parse(input_graph_path, format="turtle")
        logger.info(f"Loaded existing graph with {len(existing_graph)} triples")
    except Exception as e:
        logger.error(f"Error loading existing graph: {e}")
        return
    
    # Save the original existing graph for reference
    existing_graph_path = os.path.join(pipeline_dir, "original_graph.ttl")
    existing_graph.serialize(existing_graph_path, format="turtle")
    logger.info(f"Saved original existing graph to {existing_graph_path}")
    
    # Extract base terms from the graph
    base_terms = extract_base_terms_from_graph(existing_graph)
    
    print("\nExtracted Base Terms:")
    for term in base_terms:
        print(f"- {term}")
    
    # Extract schematic relations
    schematic_relations = extract_schematic_relations(existing_graph)
    
    print(f"\nExtracted {len(schematic_relations)} schematic relations")
    
    # Process each schematic relation
    all_verbalizations = []
    
    # Check if test_text is provided
    if test_text:
        # Use the test text directly, skipping Ollama
        logger.info(f"Using provided test text: '{test_text}'")
        cleaned_text = test_text
    else:
        # Process each relation with appropriate model
        for i, relation_graph in enumerate(schematic_relations):
            logger.info(f"=========== PROCESSING RELATION {i} ===========")
            # Serialize the relation graph
            relation_ttl = serialize_graph_as_turtle(relation_graph)
            
            # Save relation graph as string for inspection
            relation_ttl_file = os.path.join(pipeline_dir, f"relation_{i}_turtle.ttl")
            with open(relation_ttl_file, "w") as f:
                f.write(relation_ttl)
            logger.info(f"Saved relation Turtle content to: {relation_ttl_file}")
            
            # Get verbalization based on provided arguments
            if args.online_model and args.api_key:
                logger.info(f"Using online model: {args.online_model}")
                verbalization = get_verbalization_from_anthropic(
                    relation_ttl, 
                    model=args.online_model,
                    api_key=args.api_key,
                    output_dir=pipeline_dir
                )
            else:
                logger.info(f"Using Ollama model: {args.model}")
                verbalization = get_verbalization_from_ollama(
                    relation_ttl, 
                    model=args.model, 
                    output_dir=pipeline_dir
                )
            
            # Save relation verbalization to a dedicated file
            relation_verbalization_file = os.path.join(pipeline_dir, f"relation_{i}_verbalization.txt")
            with open(relation_verbalization_file, "w") as f:
                f.write(verbalization)
            
            # Save to list
            all_verbalizations.append(verbalization)
            logger.info(f"Added verbalization to list: '{verbalization}'")
            
            # Save the relation graph for reference
            relation_path = os.path.join(pipeline_dir, f"relation_{i}.ttl")
            relation_graph.serialize(relation_path, format="turtle")
            logger.info(f"Saved relation {i} to {relation_path}")
        
        # Combine verbalizations
        combined_text = " ".join(all_verbalizations)
        logger.info(f"=========== COMBINED TEXT FOR AMR PARSING ===========")
        
        # Clean up the combined text
        cleaned_text = re.sub(r'^\s*\w+:\w+\s+', '', combined_text)
        cleaned_text = re.sub(r'^answer\s+', '', cleaned_text, flags=re.IGNORECASE)
        
        logger.info(f"Original text: '{combined_text}'")
        logger.info(f"Cleaned text for AMR parsing: '{cleaned_text}'")
    
    # Save the text to be used for AMR parsing
    combined_text_file = os.path.join(pipeline_dir, "combined_text.txt")
    with open(combined_text_file, "w") as f:
        f.write(cleaned_text)
    logger.info(f"Saved text for AMR parsing to {combined_text_file}")
    
    # Load SPRING model if requested
    spring_model = None
    if args.use_spring:
        spring_model = load_spring_model(spring_model_file)
    
    # Use SPRING AMR parser or fall back to simple parser
    if args.use_spring and spring_model is not None:
        logger.info(f"Using SPRING AMR parser with model: {spring_model_file}")
        amr_representation = get_amr_from_spring_parser(cleaned_text, spring_model)
    else:
        if args.use_spring:
            logger.error("SPRING model loading failed. Cannot proceed without a valid AMR parser.")
            return
        else:
            logger.error("SPRING parser is required. Please use --use-spring flag.")
            return
    
    if amr_representation is None:
        logger.error("Failed to generate AMR representation")
        return
    
    # Create the AMR2FRED converter
    logger.info(f"Creating AMR2FRED converter")
    converter = Amr2fred()
    
    # Use AMR representation directly with AMR2FRED
    logger.info(f"=========== CALLING AMR2FRED WITH AMR REPRESENTATION ===========")
    
    # Get the newly generated graph from AMR2FRED (without merging)
    logger.info(f"Calling converter.translate with AMR representation")
    new_graph_path = os.path.join(pipeline_dir, "new_graph.ttl")
    
    try:
        # Save the newly generated graph
        new_graph_str = converter.translate(amr=amr_representation, serialize=True)
        logger.info(f"Translator returned result of type: {type(new_graph_str)}")
        
        # Save the raw output for inspection
        raw_output_file = os.path.join(pipeline_dir, "amr2fred_raw_output.txt")
        with open(raw_output_file, "wb") as f:
            if isinstance(new_graph_str, str):
                f.write(new_graph_str.encode('utf-8'))
            else:
                f.write(str(new_graph_str).encode('utf-8'))
        logger.info(f"Saved raw AMR2FRED output to {raw_output_file}")
        
        # Check if we got a valid response
        if isinstance(new_graph_str, str) and new_graph_str.startswith("Sorry, no amr"):
            logger.error(f"AMR2FRED failed to translate: {new_graph_str}")
            new_graph = Graph()
        else:
            new_graph = Graph()
            try:
                new_graph.parse(data=new_graph_str, format="turtle")
                logger.info(f"Successfully parsed AMR2FRED output into graph with {len(new_graph)} triples")
            except Exception as parse_error:
                logger.error(f"Error parsing RDF: {parse_error}")
                logger.error(f"Failed to parse AMR2FRED output")
                new_graph = Graph()
        
        # Save the new graph to file
        new_graph.serialize(new_graph_path, format="turtle")
        logger.info(f"Saved AMR2FRED graph to {new_graph_path} with {len(new_graph)} triples")
        
        # Verify the new graph was saved correctly
        if os.path.exists(new_graph_path):
            logger.info(f"Verified new graph file exists at {new_graph_path}")
            # Read the file to verify it's not empty
            with open(new_graph_path, 'r') as f:
                content = f.read()
                if not content.strip():
                    logger.error("New graph file is empty!")
                else:
                    logger.info(f"New graph file contains {len(content.splitlines())} lines")
        else:
            logger.error(f"New graph file was not created at {new_graph_path}")
        
        # Now that we have both graphs, perform the alignment and merging
        logger.info("Performing graph alignment and merging")
        merged_graph = converter.translate_with_fuzzy_mappings(
            text="",  # We don't need the text anymore as we have the graphs
            existing_graph=existing_graph,  # The graph loaded from CLI input
            base_terms=base_terms,
            alignment_method="nlp"
        )
        
        # Save the merged graph
        merged_graph_path = os.path.join(pipeline_dir, "merged_graph.ttl")
        merged_graph.serialize(merged_graph_path, format="turtle")
        logger.info(f"Saved merged graph to {merged_graph_path} with {len(merged_graph)} triples")
        
        # Generate visualizations
        save_graph_visualizations(existing_graph, "original_graph", pipeline_dir)
        save_graph_visualizations(new_graph, "new_graph", pipeline_dir)
        save_graph_visualizations(merged_graph, "merged_graph", pipeline_dir)
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        logger.exception("Detailed error information:")
        return


if __name__ == "__main__":
    main()
