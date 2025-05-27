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
from util import TafPostProcessor  # Explicitly import TafPostProcessor
#from visualization import DigraphWriter
from nltk.corpus import wordnet
import difflib
from rdflib import Namespace

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
test_text = "Inside a barred circus cage, a man—clearly the Tramp character played by Charlie Chaplin (agent)—is caught in a moment of intense comedic tension as he holds a large, flat object like a shield (instrument) between himself and a full-grown lion (theme) lying on the floor just feet away. The man's wide-eyed expression and stiff posture convey alarm and desperation as he edges backward, clearly aware of the imminent danger. Despite the threat, the lion remains completely still, its eyes closed, resting its massive head on its paws in a pose of total indifference. The man's defensive stance and panicked energy are in stark contrast to the lion's calm, creating a powerful visual irony central to the comedy."  #"""This black and white film still shows a scene from what appears to be a silent-era comedy. In the center of the frame stands a man in casual attire (vest and light-colored shirt) on a street or sidewalk. On either side of him, two uniformed police officers are positioned on the steps of what looks like a government or public building with classical columns and large doors visible in the background. The officers are in dramatic poses with their arms raised - one appears to be holding a baton or nightstick. The composition suggests the police officers are either apprehending the central figure or possibly performing some kind of comedic routine around him."""

# Ollama model to use
ollama_model = "llama3.1:8b"

# SPRING model file to use - updated to AMR3.parsing.pt
spring_model_file = "AMR3.parsing.pt"

# === Helper Functions for Entity Comparison ===
def _extract_base_name(uri):
    """Extracts the base name from a URI, removing prefixes and suffixes like _123."""
    if not isinstance(uri, str):
        return ""
    try:
        # Get the part after the last / or #
        local_name = uri.split('/')[-1].split('#')[-1]
        # Remove potential numeric suffix after underscore
        base_name = local_name.split('_')[0]
        # Handle potential prefix like ns2:
        base_name = base_name.split(':')[-1]
        return base_name.lower()
    except Exception:
        return ""

def _wordnet_similarity(word1, word2):
    """Calculate similarity between two words using WordNet path similarity."""
    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)
    max_similarity = 0.0
    if synsets1 and synsets2:
        for s1 in synsets1:
            for s2 in synsets2:
                similarity = s1.path_similarity(s2)
                if similarity is not None and similarity > max_similarity:
                    max_similarity = similarity
    return max_similarity

def _calculate_similarity_score(entity1_uri, entity2_uri):
    """Calculate a combined similarity score."""
    name1 = _extract_base_name(entity1_uri)
    name2 = _extract_base_name(entity2_uri)

    if not name1 or not name2:
        return 0.0

    # 1. String Similarity (using SequenceMatcher)
    string_sim = difflib.SequenceMatcher(None, name1, name2).ratio()

    # 2. WordNet Similarity
    wn_sim = _wordnet_similarity(name1, name2)

    # 3. Embedding Similarity (Placeholder)
    embedding_sim = 0.0 # Placeholder - Implement later if needed

    # Combine scores (example weights)
    # Adjust weights as needed
    score = (0.5 * string_sim) + (0.5 * wn_sim) + (0.0 * embedding_sim)

    # Log detailed scores
    # logger.debug(f"Comparing '{name1}' ({entity1_uri}) and '{name2}' ({entity2_uri}):")
    # logger.debug(f"  String Sim: {string_sim:.2f}, WordNet Sim: {wn_sim:.2f}, Embed Sim: {embedding_sim:.2f} -> Score: {score:.2f}")

    return score

def extract_typed_individuals(graph):
    """
    Extracts individuals from a graph based on specific type assertions.
    Returns a dictionary {uri: base_name}.
    """
    individuals = {}
    # SPARQL query to get entities that are explicitly NamedIndividuals
    # or have a type other than known property/class types.
    query = """
    SELECT DISTINCT ?individual
    WHERE {
        {
            ?individual rdf:type owl:NamedIndividual .
        } UNION {
            ?individual rdf:type ?type .
            FILTER NOT EXISTS { VALUES ?excludedType { owl:ObjectProperty owl:DatatypeProperty owl:AnnotationProperty owl:Class } ?individual rdf:type ?excludedType . }
            FILTER (!isBlank(?individual))
        }
    }
    """
    try:
        results = graph.query(query)
        for row in results:
            uri = str(row.individual)
            base_name = _extract_base_name(uri)
            if base_name:
                individuals[uri] = base_name
    except Exception as e:
        logger.error(f"Error executing SPARQL query for individuals: {e}")
    logger.info(f"Extracted {len(individuals)} individuals from graph.")
    return individuals

def bind_meaningful_prefixes(graph):
    """
    Binds meaningful prefixes to a graph for better readability in serialized output.
    """
    from rdflib import Namespace
    
    # Define meaningful prefixes
    prefixes = {
        'kh': Namespace('file://./log.owl#'),
        'pblr': Namespace('https://w3id.org/framester/data/propbank-3.4.0/LocalRole/'),
        'fschema': Namespace('https://w3id.org/framester/schema/'),
        'pb': Namespace('https://w3id.org/framester/data/propbank-3.4.0/RoleSet/'),
        'fred': Namespace('http://www.ontologydesignpatterns.org/ont/fred/domain.owl#'),
        'verbatlas': Namespace('http://verbatlas.org/'),
        'dul': Namespace('http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#'),
        'rdf': Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#'),
        'owl': Namespace('http://www.w3.org/2002/07/owl#'),
        'rdfs': Namespace('http://www.w3.org/2000/01/rdf-schema#'),
        'xsd': Namespace('http://www.w3.org/2001/XMLSchema#')
    }
    
    # Bind each prefix to the graph
    for prefix, namespace in prefixes.items():
        graph.bind(prefix, namespace)
    
    return graph

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
    
    # Create a clearer prompt that emphasizes responding with just the verbalization
    prompt = f"""
You receive some Turtle/RDF triples describing a relation between objects, and you need to generate a natural language verbalization.

Here's an example:

Triples:
ns2:schematicRelation_14 a ns2:Occlusion,
        owl:NamedIndividual ;
    ns2:eventMode ns2:Ended ;
    ns2:hasOccludee ns2:knife_70 ;
    ns2:hasOccluder ns2:apple_27 .

Good verbalization: "The apple is occluding the knife."

Now, for the following triples, respond with ONLY a simple, clear verbalization that describes the relationship.
Disregard any numbers in object identifiers (like _70) and focus on the relationship type and the objects involved.
Don't include any labels, quotes, or other formatting - just the plain verbalization sentence.

Triples:
{triples}
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
        ollama_output_file = os.path.join(pipeline_dir, f"{call_id}_response.txt")
        with open(ollama_output_file, "w") as f:
            f.write(full_response)
            
        logger.info(f"Saved full Ollama response to: {ollama_output_file}")
        
        # Clean up the response
        # Try to extract the actual verbalization, handling various possible formats
        clean_response = full_response.strip()
        
        # Remove any "Verbalization:" prefix
        clean_response = re.sub(r'^["\']?Verbalization:?["\']?\s*', '', clean_response, flags=re.IGNORECASE)
        
        # Remove any quotes
        clean_response = re.sub(r'^["\']|["\']$', '', clean_response)
        
        # If we still have the word "verbalization" by itself, it's a bad response
        if clean_response.lower() == "verbalization":
            logger.warning("Invalid response: just the word 'Verbalization'")
            return _get_fallback_verbalization(triples)
            
        # If we have a valid verbalization, return it
        if clean_response and not clean_response.isspace():
            logger.info(f"Extracted verbalization: '{clean_response}'")
            return clean_response
        else:
            # If verbalization is empty or just whitespace, use fallback generator
            logger.warning("Empty verbalization received from Ollama, using fallback generator")
            return _get_fallback_verbalization(triples)
            
    except Exception as e:
        error_msg = f"Exception when calling Ollama: {e}"
        logger.error(error_msg)
        logger.info("Using fallback generator for verbalization")
        return _get_fallback_verbalization(triples)


def _get_fallback_verbalization(triples):
    """
    Generate a more meaningful fallback verbalization based on the triples
    """
    try:
        logger.info("Generating fallback verbalization from triples")
        # Look for specific patterns in the triples
        subject_types = {}
        predicates = {}
        
        # Basic parsing of turtle format
        for line in triples.splitlines():
            line = line.strip()
            if not line or line.startswith('@prefix') or line.startswith('#'):
                continue
            
            # Try to identify types and predicates
            if 'a ' in line:
                parts = line.split('a ')
                if len(parts) >= 2:
                    subject = parts[0].strip()
                    type_name = parts[1].strip().rstrip(' ;.')
                    if type_name.endswith('Occlusion'):
                        subject_types[subject] = 'Occlusion'
                    elif type_name.endswith('Penetration'):
                        subject_types[subject] = 'Penetration'
                    elif type_name.endswith('Contact'):
                        subject_types[subject] = 'Contact'
                    elif type_name.endswith('Support'):
                        subject_types[subject] = 'Support'
                    else:
                        subject_types[subject] = type_name
            
            # Look for specific predicates
            for predicate in ['hasOccluder', 'hasOccludee', 'hasPenetrator', 'hasPenetratee']:
                if predicate in line:
                    parts = line.split(predicate)
                    if len(parts) >= 2:
                        subject = parts[0].strip()
                        object_val = parts[1].strip().rstrip(' ;.')
                        if subject not in predicates:
                            predicates[subject] = []
                        predicates[subject].append((predicate, object_val))
        
        # Generate a better verbalization based on types and predicates
        for subject, type_name in subject_types.items():
            if subject in predicates:
                if type_name == 'Occlusion':
                    # Find occluder and occludee
                    occluder = None
                    occludee = None
                    for pred, obj in predicates[subject]:
                        if pred == 'hasOccluder':
                            occluder = _clean_entity_name(obj)
                        elif pred == 'hasOccludee':
                            occludee = _clean_entity_name(obj)
                    if occluder and occludee:
                        return f"The {occluder} is occluding the {occludee}."
                
                elif type_name == 'Penetration':
                    # Find penetrator and penetratee
                    penetrator = None
                    penetratee = None
                    for pred, obj in predicates[subject]:
                        if pred == 'hasPenetrator':
                            penetrator = _clean_entity_name(obj)
                        elif pred == 'hasPenetratee':
                            penetratee = _clean_entity_name(obj)
                    if penetrator and penetratee:
                        return f"The {penetrator} is penetrating the {penetratee}."
        
        # If we couldn't generate anything specific
        logger.warning("Could not generate specific fallback verbalization, using generic fallback")
        return "An object is interacting with another object."
        
    except Exception as e:
        logger.error(f"Error in fallback verbalization: {e}")
        return "An object is interacting with another object."


def _clean_entity_name(entity_uri):
    """Extract a clean name from an entity URI"""
    try:
        # Extract the local name (last part of URI)
        local_name = entity_uri.split('/')[-1].split('#')[-1].strip()
        
        # Remove any numeric suffix
        base_name = local_name.split('_')[0] if '_' in local_name else local_name
        
        # Replace camelCase or PascalCase with spaces
        spaced_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', base_name).lower()
        
        return spaced_name
    except Exception:
        return "object"


def get_verbalization_from_anthropic(triples, model="claude-3-7-sonnet-20250219", api_key=None, output_dir="./out/anthropic/"):
    """
    Get verbalization from Anthropic's Claude API for the given triples
    """
    # Create a TafPostProcessor to use its verbalization method
    processor = TafPostProcessor()
    return processor.get_verbalization_from_anthropic(triples, model, api_key, output_dir)


def serialize_graph_as_turtle(graph):
    """
    Serialize a graph as turtle with meaningful prefixes, returning the string representation.
    """
    # Apply meaningful prefixes before serialization
    graph = bind_meaningful_prefixes(graph)
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
        spring_amr_path = os.path.join(implementation_dir, "spring_khafre", "spring_amr")
        
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


def enrich_graph_with_claude(graph, text, api_key, output_dir="./pipeline_out/"):
    """
    Enrich a knowledge graph with implicit knowledge using Claude API.
    
    Args:
        graph: The RDF graph to enrich
        text: The original text that generated the graph
        api_key: Claude API key
        output_dir: Directory to save outputs
        
    Returns:
        The enriched graph
    """
    logger.info("=========== ENRICHING GRAPH WITH CLAUDE ===========")
    print("\n=========== ENRICHING GRAPH WITH CLAUDE ===========")
    print(f"Processing graph with {len(graph)} triples")
    
    # Initialize enrichment statistics
    stats = {
        "implied_events": 0,
        "dangerous_events": 0,
        "danger_avoidance": 0,
        "risk_reduction": 0,
        "other_triples": 0,
        "total_new_triples": 0
    }
    
    # Serialize the graph to Turtle format
    graph_ttl = serialize_graph_as_turtle(graph)
    
    # Create the prompt with the graph and text
    prompt = f"""
Task:
Your goal is to extend a knowledge graph KG with more knowledge that can be assumed from a text T, but it is not explicit.
Using the elements of KG, as well as PropBank, WordNet and other graph elements as anchoring points, add any further elements you need to extract implicit knowledge about:
1. Implied Future Events
2. Potentially Dangerous Implied Future Events
3. Danger Avoidance Actions
4. Risk Reduction Reactions

IMPORTANT: Your response MUST start with the following prefix declarations:
@prefix kh: <file://./log.owl#> .
@prefix pblr: <https://w3id.org/framester/data/propbank-3.4.0/LocalRole/> .
@prefix fschema: <https://w3id.org/framester/schema/> .
@prefix pb: <https://w3id.org/framester/data/propbank-3.4.0/RoleSet/> .
@prefix fred: <http://www.ontologydesignpatterns.org/ont/fred/domain.owl#> .
@prefix verbatlas: <http://verbatlas.org/> .
@prefix dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix en: <file://./log_enriched.owl#> .

If you add further prefixes, make sure to declare them.

Start with Implied Future Events. An implied future event happens when two events mentioned in a text have a causal dependence sequence; here are natural language inference examples:
1) The car driver didn't notice the traffic light turns red. -> The car driver won't stop at the traffic light.
Here is the rough triplification, you will be more precise and use KG nodes, PropBank, WordNet, and WikiData entities as anchor point:
1) en:carDriver en:notStoppingAt en:TrafficLight .
en:CarDriverStoppingAtTrafficLight a dul:Situation ; en:hasValue "False"^^xsd:boolean .
en:CarDriverStoppingAtTrafficLight en:hasParticipant en:CarDriver .
en:CarDriverStoppingAtTrafficLight en:hasParticipant en:TrafficLight .

Add Implied Future Event triples to the KG considering the original text T.

Proceed with Potentially Dangerous Implied Future Events. In some cases, an implied future event can be dangerous. Here are natural language inference examples:
1) The car does not stop at traffic lights. There are pedestrian in the crosswalk. --> The people could be hit by the car.
2) After checking the arrival of cars, the pedestrian is crossing on the crosswalk.
Here is the rough triplification, you will be more precise and use KG nodes, PropBank, WordNet, and WikiData entities as anchor point:
1) en:CarNotStopping en:introducesDanger "true"^^xsd:boolean .
2) en:PedestrianCrossingOnCrosswalk en:isDangerous "false"^^xsd:boolean .

Add Potentially Dangerous Implied Future Events triples to the KG considering the original text T and the Potentially Dangerous Implied Future Events.

Add now Danger Avoidance Actions, namely the actions that should have happened to avoid the danger. 
Here are natural language inference examples:
1) The car does not stop at traffic lights. There are pedestrians in the crosswalk. --> The car should have stopped at the traffic light.
Here is the rough triplification, you will be more precise and use KG nodes, PropBank, WordNet, and WikiData entities as anchor point:
1) en:CarStopping en:isDangerAvoidanceAction "true"^^xsd:boolean .

Add Danger Avoidance Actions triples to the KG considering the original text T and the Potentially Dangerous Implied Future Events.

Finally, add Risk Reduction Reaction, namely the reactions that could happen, once the danger exists, to reduce the impact of the danger. Here are natural language inference examples:
1) The car does not stop at traffic lights. There are pedestrians in the crosswalk. --> The pedestrians could quickly jump back to avoid the car.
Here is the rough triplification, you will be more precise and use KG nodes, PropBank, WordNet, and WikiData entities as anchor point:
1) en:PedestrianJumpingBack a dul:Situation ; en:hasRiskReductionValue 0.5 .
en:PedestrianJumpingBack en:hasParticipant en:pedestrian .

Add Risk Reduction Reactions triples to the KG considering the original text T and the Potentially Dangerous Implied Future Events.

IMPORTANT: Print the newly generater triples only. Do not write anything else.
If you introduce Datatype and Object properties add a triple declaring them as such.

T:
{text}

KG:
{graph_ttl}
"""
    
    # Create a unique identifier for this call
    call_id = f"claude_enrichment_{int(time.time())}"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Log the prompt to a file
    prompt_file = os.path.join(output_dir, f"{call_id}_prompt.txt")
    with open(prompt_file, "w") as f:
        f.write(prompt)
    
    logger.info(f"Prompt saved to: {prompt_file}")
    print(f"Prompt saved to: {prompt_file}")
    
    try:
        # Check if anthropic package is available
        try:
            import anthropic
            logger.info("Anthropic package is available")
            print("Anthropic package is available")
        except ImportError:
            error_msg = "ERROR: Anthropic package is NOT installed. Please run: pip install anthropic"
            logger.error(error_msg)
            print(error_msg)
            return graph
        
        # Initialize Anthropic client
        if not api_key:
            error_msg = "No API key provided for Anthropic"
            logger.error(error_msg)
            print(error_msg)
            return graph
            
        client = anthropic.Anthropic(api_key=api_key)
        
        # Get list of models to try in order
        models_to_try = [
            "claude-3-7-sonnet-20250219",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
        
        # Try each model until one works
        content = None
        used_model = None
        
        print("\nAttempting to connect to Claude API...")
        for current_model in models_to_try:
            try:
                logger.info(f"Trying to use model: {current_model}")
                print(f"Trying to use model: {current_model}")
                
                # Call the API
                message = client.messages.create(
                    model=current_model,
                    max_tokens=4000,
                    system="You are a specialized assistant that extends knowledge graphs with implicit knowledge. Respond ONLY with the new triples in Turtle format.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                if message and hasattr(message, 'content'):
                    content = message.content[0].text
                    used_model = current_model
                    success_msg = f"Successfully used model: {used_model}"
                    logger.info(success_msg)
                    print(success_msg)
                    break
                else:
                    warning_msg = f"Model {current_model} returned unexpected response format"
                    logger.warning(warning_msg)
                    print(warning_msg)
                    
            except Exception as e:
                warning_msg = f"Failed to use model {current_model}: {e}"
                logger.warning(warning_msg)
                print(warning_msg)
                continue
        
        # If no model worked, return original graph
        if content is None:
            error_msg = "All Claude models failed"
            logger.error(error_msg)
            print(error_msg)
            return graph
        
        # Save the full response to a file
        response_file = os.path.join(output_dir, f"{call_id}_response.txt")
        with open(response_file, "w") as f:
            f.write(content)
            
        logger.info(f"Saved Claude response to: {response_file}")
        print(f"Saved Claude response to: {response_file}")
        
        # Create a new graph for the enriched triples
        enriched_graph = Graph()
        
        try:
            # Clean the response before parsing
            cleaned_content = content.strip()
            
            # Remove any binary characters and other unwanted prefixes
            cleaned_content = re.sub(r'^b[\'"]', '', cleaned_content)  # Remove b' or b" at start
            cleaned_content = re.sub(r'[\'"]$', '', cleaned_content)   # Remove ' or " at end
            cleaned_content = re.sub(r'\\n', '\n', cleaned_content)    # Convert \n to actual newlines
            cleaned_content = re.sub(r'\\r', '', cleaned_content)      # Remove \r
            cleaned_content = re.sub(r'\\t', '    ', cleaned_content)  # Convert \t to spaces
            
            # Remove any non-Turtle content (like markdown code blocks)
            cleaned_content = re.sub(r'```turtle\n?', '', cleaned_content)
            cleaned_content = re.sub(r'```\n?', '', cleaned_content)
            
            # Remove any empty lines at start/end
            cleaned_content = cleaned_content.strip()
            
            # Save the cleaned content for debugging
            cleaned_response_file = os.path.join(output_dir, f"{call_id}_cleaned_response.txt")
            with open(cleaned_response_file, "w") as f:
                f.write(cleaned_content)
            logger.info(f"Saved cleaned response to: {cleaned_response_file}")
            print(f"Saved cleaned response to: {cleaned_response_file}")
            
            # Parse the cleaned triples into the enriched graph
            print("\nParsing enrichment triples...")
            enriched_graph.parse(data=cleaned_content, format="turtle")
            
            # Count triples by category
            print("Analyzing enriched triples by category...")
            
            # Count Implied Future Events
            implied_events_query = """
            SELECT (COUNT(?s) as ?count)
            WHERE {
                ?s ?p ?o .
                FILTER(CONTAINS(STR(?s), "ImpliedFuture") || CONTAINS(STR(?p), "implied") || 
                       CONTAINS(STR(?o), "ImpliedFuture") || CONTAINS(STR(?s), "FutureEvent"))
            }
            """
            
            # Count Dangerous Events
            dangerous_events_query = """
            SELECT (COUNT(?s) as ?count)
            WHERE {
                { ?s ?p ?o . 
                  FILTER(CONTAINS(STR(?s), "Dangerous") || CONTAINS(STR(?p), "danger") || 
                         CONTAINS(STR(?o), "Dangerous") || CONTAINS(STR(?p), "isDangerous")) }
                UNION
                { ?s ?p ?o .
                  FILTER(CONTAINS(STR(?p), "introducesDanger")) }
            }
            """
            
            # Count Danger Avoidance Actions
            avoidance_query = """
            SELECT (COUNT(?s) as ?count)
            WHERE {
                { ?s ?p ?o . 
                  FILTER(CONTAINS(STR(?s), "Avoidance") || CONTAINS(STR(?p), "avoid") || 
                         CONTAINS(STR(?o), "Avoidance")) }
                UNION
                { ?s ?p ?o .
                  FILTER(CONTAINS(STR(?p), "isDangerAvoidanceAction")) }
            }
            """
            
            # Count Risk Reduction Reactions
            risk_reduction_query = """
            SELECT (COUNT(?s) as ?count)
            WHERE {
                { ?s ?p ?o . 
                  FILTER(CONTAINS(STR(?s), "Reduction") || CONTAINS(STR(?p), "reduce") || 
                         CONTAINS(STR(?o), "Reduction") || CONTAINS(STR(?p), "hasRiskReduction")) }
                UNION
                { ?s ?p ?o .
                  FILTER(CONTAINS(STR(?p), "hasRiskReductionValue")) }
            }
            """
            
            # Execute the queries
            try:
                stats["implied_events"] = list(enriched_graph.query(implied_events_query))[0][0].value
                stats["dangerous_events"] = list(enriched_graph.query(dangerous_events_query))[0][0].value
                stats["danger_avoidance"] = list(enriched_graph.query(avoidance_query))[0][0].value
                stats["risk_reduction"] = list(enriched_graph.query(risk_reduction_query))[0][0].value
                stats["total_new_triples"] = len(enriched_graph)
                stats["other_triples"] = (stats["total_new_triples"] - stats["implied_events"] 
                                         - stats["dangerous_events"] - stats["danger_avoidance"] 
                                         - stats["risk_reduction"])
                if stats["other_triples"] < 0:
                    stats["other_triples"] = 0  # Avoid negative count due to overlaps
            except Exception as e:
                logger.error(f"Error counting triples by category: {e}")
                print(f"Error counting triples by category: {e}")
            
            # Print counts to console
            print(f"\n=== ENRICHMENT STATISTICS ===")
            print(f"Total new triples added: {stats['total_new_triples']}")
            print(f"Implied Future Events: {stats['implied_events']}")
            print(f"Potentially Dangerous Events: {stats['dangerous_events']}")
            print(f"Danger Avoidance Actions: {stats['danger_avoidance']}")
            print(f"Risk Reduction Reactions: {stats['risk_reduction']}")
            print(f"Other triples: {stats['other_triples']}")
            print(f"===========================")
            
            logger.info(f"Successfully parsed {len(enriched_graph)} new triples from Claude response")
            
            # Generate a human-readable summary of the enrichment
            try:
                # Extract some example triples for each category to include in the summary
                examples = {
                    "implied_events": [],
                    "dangerous_events": [],
                    "danger_avoidance": [],
                    "risk_reduction": []
                }
                
                # Helper function to extract a human-readable version of a triple
                def readable_triple(s, p, o):
                    s_str = str(s).split('/')[-1].split('#')[-1]
                    p_str = str(p).split('/')[-1].split('#')[-1]
                    if isinstance(o, str) or not hasattr(o, '__str__'):
                        o_str = str(o)
                    else:
                        o_str = str(o).split('/')[-1].split('#')[-1]
                    return f"{s_str} {p_str} {o_str}"
                
                # Get examples for each category (up to 3 per category)
                for s, p, o in enriched_graph:
                    triple_str = readable_triple(s, p, o)
                    
                    # Check which category this triple belongs to
                    if (len(examples["implied_events"]) < 3 and 
                        ("ImpliedFuture" in str(s) or "implied" in str(p) or 
                         "ImpliedFuture" in str(o) or "FutureEvent" in str(s))):
                        examples["implied_events"].append(triple_str)
                    
                    elif (len(examples["dangerous_events"]) < 3 and 
                          ("Dangerous" in str(s) or "danger" in str(p) or 
                           "Dangerous" in str(o) or "isDangerous" in str(p) or
                           "introducesDanger" in str(p))):
                        examples["dangerous_events"].append(triple_str)
                    
                    elif (len(examples["danger_avoidance"]) < 3 and 
                          ("Avoidance" in str(s) or "avoid" in str(p) or 
                           "Avoidance" in str(o) or "isDangerAvoidanceAction" in str(p))):
                        examples["danger_avoidance"].append(triple_str)
                    
                    elif (len(examples["risk_reduction"]) < 3 and 
                          ("Reduction" in str(s) or "reduce" in str(p) or 
                           "Reduction" in str(o) or "hasRiskReduction" in str(p) or
                           "hasRiskReductionValue" in str(p))):
                        examples["risk_reduction"].append(triple_str)
            except Exception as e:
                logger.error(f"Error extracting example triples: {e}")
                print(f"Error extracting example triples: {e}")
                # Initialize with empty lists
                examples = {k: [] for k in examples}
            
            # Create a summary file
            summary_file = os.path.join(output_dir, "enrichment_summary.txt")
            with open(summary_file, "w") as f:
                f.write("===== GRAPH ENRICHMENT SUMMARY =====\n\n")
                f.write(f"Original text: \"{text}\"\n\n")
                f.write(f"Enrichment performed using model: {used_model}\n")
                f.write(f"Original graph had {len(graph)} triples\n")
                f.write(f"Added {len(enriched_graph)} new triples\n\n")
                
                f.write("--- STATISTICS ---\n")
                f.write(f"Implied Future Events: {stats['implied_events']}\n")
                f.write(f"Potentially Dangerous Events: {stats['dangerous_events']}\n")
                f.write(f"Danger Avoidance Actions: {stats['danger_avoidance']}\n")
                f.write(f"Risk Reduction Reactions: {stats['risk_reduction']}\n")
                f.write(f"Other triples: {stats['other_triples']}\n\n")
                
                f.write("--- EXAMPLE TRIPLES ---\n")
                f.write("Implied Future Events:\n")
                for ex in examples["implied_events"]:
                    f.write(f"  - {ex}\n")
                f.write("\nPotentially Dangerous Events:\n")
                for ex in examples["dangerous_events"]:
                    f.write(f"  - {ex}\n")
                f.write("\nDanger Avoidance Actions:\n")
                for ex in examples["danger_avoidance"]:
                    f.write(f"  - {ex}\n")
                f.write("\nRisk Reduction Reactions:\n")
                for ex in examples["risk_reduction"]:
                    f.write(f"  - {ex}\n")
                
                f.write("\n--- FULL ENRICHMENT ---\n")
                f.write("For complete results, see the enriched graph file and cleaned_response.txt\n")
            
            logger.info(f"Created enrichment summary at {summary_file}")
            print(f"\nCreated enrichment summary at {summary_file}")
            
            # Merge the new triples with the original graph
            print("Merging enriched triples with original graph...")
            graph += enriched_graph
            
            # Save the enriched graph
            enriched_graph_path = os.path.join(output_dir, "enriched_graph.ttl")
            graph.serialize(enriched_graph_path, format="turtle")
            logger.info(f"Saved enriched graph to {enriched_graph_path}")
            print(f"Saved enriched graph to {enriched_graph_path}")
            
            print("\n==================================================")
            print(f"ENRICHMENT COMPLETE: Added {len(enriched_graph)} new triples to the graph")
            print("==================================================")
            
            return graph
            
        except Exception as e:
            error_msg = f"Error parsing Claude response as triples: {e}"
            logger.error(error_msg)
            # Log the problematic content for debugging
            logger.error(f"Problematic content: {content[:200]}...")  # Log first 200 chars
            print(error_msg)
            print(f"Problematic content (first 200 chars): {content[:200]}...")
            return graph
            
    except Exception as e:
        error_msg = f"Error in graph enrichment: {e}"
        logger.error(error_msg)
        print(error_msg)
        return graph


def ground_enrichment(graph, text, output_dir="./pipeline_out/"):
    """
    Enrich a knowledge graph by grounding complex situations with action/motion types.
    
    Args:
        graph: The RDF graph to enrich
        text: The original text that generated the graph
        output_dir: Directory to save outputs
        
    Returns:
        The grounded graph
    """
    logger.info("=========== GROUNDING ENRICHMENT ===========")
    print("\n=========== GROUNDING ENRICHMENT ===========")
    
    # Create a new graph for the grounded triples
    grounded_graph = Graph()
    
    # Bind the recommender system namespace
    rs = Namespace('file://./log_recommender_system.owl#')
    grounded_graph.bind('rs', rs)
    
    # Bind our own DUL namespace
    DUL = Namespace('http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#')
    grounded_graph.bind('dul', DUL)
    
    # Find all dul:Situation nodes with 'en' prefix
    query = """
    SELECT DISTINCT ?situation ?participant
    WHERE {
        ?situation a dul:Situation .
        FILTER(CONTAINS(str(?situation), "en:"))
        OPTIONAL {
            ?situation ?p ?participant .
            FILTER(CONTAINS(str(?p), "hasParticipant"))
        }
    }
    """
    
    try:
        results = graph.query(query)
        situations = {}
        
        # Group participants by situation
        for row in results:
            situation = row.situation
            participant = row.participant
            if situation not in situations:
                situations[situation] = []
            if participant:
                situations[situation].append(participant)
        
        # Process each situation
        for situation, participants in situations.items():
            # Add grounding triples based on the situation type
            situation_str = str(situation)
            
            # Check for dangerous situations
            if "Dangerous" in situation_str:
                grounded_graph.add((situation, rs.requires, rs.Contact))
                # Add roles for participants
                for participant in participants:
                    if "Avoidance" in situation_str:
                        grounded_graph.add((participant, DUL.hasRole, rs.Contacter))
                    else:
                        grounded_graph.add((participant, DUL.hasRole, rs.Contacted))
            
            # Check for avoidance actions
            elif "Avoidance" in situation_str:
                grounded_graph.add((situation, rs.requires, rs.Movement))
                grounded_graph.add((situation, rs.requires, rs.Departure))
                # Add roles for participants
                for participant in participants:
                    grounded_graph.add((participant, DUL.hasRole, rs.Departer))
            
            # Check for risk reduction reactions
            elif "Reduction" in situation_str:
                grounded_graph.add((situation, rs.requires, rs.Movement))
                grounded_graph.add((situation, rs.requires, rs.Approach))
                # Add roles for participants
                for participant in participants:
                    grounded_graph.add((participant, DUL.hasRole, rs.Approacher))
            
            # Check for implied future events
            elif "Future" in situation_str:
                grounded_graph.add((situation, rs.requires, rs.Movement))
                # Add roles for participants
                for participant in participants:
                    grounded_graph.add((participant, DUL.hasRole, rs.Trajector))
        
        # Save the grounded graph
        grounded_graph_path = os.path.join(output_dir, "enriched_grounded_graph.ttl")
        grounded_graph.serialize(grounded_graph_path, format="turtle")
        logger.info(f"Saved grounded graph to {grounded_graph_path}")
        print(f"Saved grounded graph to {grounded_graph_path}")
        
        # Generate visualization
        save_graph_visualizations(grounded_graph, "enriched_grounded_graph", output_dir)
        
        # Create a summary file
        summary_file = os.path.join(output_dir, "grounding_summary.txt")
        with open(summary_file, "w") as f:
            f.write("===== GROUNDING ENRICHMENT SUMMARY =====\n\n")
            f.write(f"Original text: \"{text}\"\n\n")
            f.write(f"Processed {len(situations)} situations\n")
            f.write(f"Added {len(grounded_graph)} new triples\n\n")
            
            f.write("--- GROUNDED SITUATIONS ---\n")
            for situation, participants in situations.items():
                f.write(f"\nSituation: {situation}\n")
                f.write("Participants:\n")
                for participant in participants:
                    f.write(f"  - {participant}\n")
                # Get the grounding triples for this situation
                for s, p, o in grounded_graph.triples((situation, None, None)):
                    f.write(f"  Grounding: {p} {o}\n")
        
        logger.info(f"Created grounding summary at {summary_file}")
        print(f"\nCreated grounding summary at {summary_file}")
        
        # Merge the grounded triples with the original graph
        graph += grounded_graph
        
        return graph
        
    except Exception as e:
        error_msg = f"Error in grounding enrichment: {e}"
        logger.error(error_msg)
        print(error_msg)
        return graph


def main():
    # Global declarations
    global ollama_model
    global spring_model_file
    global test_text
    
    # Debug: check if test_text is empty
    logger.info(f"test_text is: '{test_text}'")
    logger.info(f"test_text is empty: {test_text == ''}")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process RDF graphs with AMR2FRED and Ollama using SPRING AMR parser")
    
    # Input/Output arguments
    parser.add_argument("--input", "-i", type=str, help="Path to input graph file", required=True)
    
    # SPRING parser arguments
    parser.add_argument("--use-spring", "-s", action="store_true", help="Use SPRING parser instead of simple AMR generator")
    parser.add_argument("--spring-model", type=str, default=spring_model_file, help="SPRING model file to use")
    
    # Verbalization arguments
    verbalization_group = parser.add_mutually_exclusive_group()
    verbalization_group.add_argument("--model", "-m", type=str, default="llama3.1:8b", 
                                   help="Ollama model to use for verbalization")
    verbalization_group.add_argument("--verbalization-model", type=str, 
                                   help="Online model to use for verbalization (e.g. claude-3-7-sonnet-20250219)")
    parser.add_argument("--verbalization-api-key", type=str, 
                       help="API key for verbalization model (required if using --verbalization-model)")
    
    # Entity alignment arguments
    parser.add_argument("--alignment-model", type=str, 
                       help="Model to use for entity alignment (e.g. claude-3-7-sonnet-20250219)")
    parser.add_argument("--alignment-api-key", type=str, 
                       help="API key for alignment model (required if using --alignment-model)")
    
    # Graph enrichment arguments
    parser.add_argument("--enrich", action="store_true", 
                       help="Enable graph enrichment with implicit knowledge")
    parser.add_argument("--enrich-model", type=str, default="claude-3-7-sonnet-20250219",
                       help="Model to use for graph enrichment")
    parser.add_argument("--enrich-api-key", type=str, 
                       help="API key for enrichment model (required if using --enrich)")
    
    # Grounding enrichment arguments
    parser.add_argument("--ground", action="store_true",
                       help="Enable grounding enrichment of situations with action/motion types")
    
    # Online services argument
    parser.add_argument("--online-services", "-o", action="store_true", 
                       help="Enable online services for better entity disambiguation (WordNet, Framester, Wikidata)")
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.verbalization_model and not args.verbalization_api_key:
        parser.error("--verbalization-api-key is required when using --verbalization-model")
    
    if args.alignment_model and not args.alignment_api_key:
        parser.error("--alignment-api-key is required when using --alignment-model")
    
    if args.enrich and not args.enrich_api_key:
        parser.error("--enrich-api-key is required when using --enrich")
    
    # Important: Set a flag to indicate whether to use online verbalization
    # Only use online verbalization when EXPLICITLY required
    args.use_online_verbalization = bool(args.verbalization_model and args.verbalization_api_key)
    
    # CRITICAL FIX: Force to use model for verbalization when --model is provided (Ollama)
    # This ensures that if you provide --model, it will be used for verbalization
    if args.model:
        args.use_online_verbalization = False
        logger.info(f"Will use local Ollama model ({args.model}) for verbalization")
        if args.verbalization_model:
            logger.warning(f"Ignoring --verbalization-model because --model is provided")
    
    # Initialize the verbalization variables
    combined_text = ""
    
    # Initialize cleaned text variable
    cleaned_text = ""
    
    # Initialize actual verbalization variable
    actual_verbalization = ""
    
    # Initialize verbalizations list
    all_verbalizations = []
    
    # DEBUG: Print all arguments
    logger.info("==== COMMAND LINE ARGUMENTS ====")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Use SPRING: {args.use_spring}")
    logger.info(f"SPRING model: {args.spring_model}")
    logger.info(f"Verbalization method: {'Online' if args.use_online_verbalization else 'Ollama'}")
    logger.info(f"Verbalization model: {args.verbalization_model if args.use_online_verbalization else args.model}")
    logger.info(f"Alignment model: {args.alignment_model}")
    logger.info(f"Enrichment enabled: {args.enrich}")
    logger.info(f"Enrichment model: {args.enrich_model}")
    logger.info(f"Online services: {args.online_services}")
    logger.info("=============================")
    
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
        # Read the file in binary mode first
        with open(input_graph_path, 'rb') as f:
            content = f.read()
        
        # Decode with UTF-8 and handle line endings
        content = content.decode('utf-8')
        content = content.replace('\r\n', '\n')  # Convert Windows line endings to Unix
        
        # Clean up binary string literals
        content = content.replace("b'", "'")  # Remove b' prefix
        content = content.replace('b"', '"')  # Remove b" prefix
        content = re.sub(r"'b'", "'", content)  # Remove 'b' in quotes
        content = re.sub(r'"b"', '"', content)  # Remove "b" in quotes
        
        # Parse the content
        existing_graph.parse(data=content, format="turtle")
        logger.info(f"Loaded existing graph with {len(existing_graph)} triples")
    except Exception as e:
        logger.error(f"Error loading existing graph: {e}")
        # Log the problematic content for debugging
        logger.error(f"Problematic content (first 200 chars): {content[:200]}")
        return
    
    # Save all three graphs for separate viewing
    original_graph_path = os.path.join(pipeline_dir, "original_graph.ttl")
    # Apply meaningful prefixes before serialization
    existing_graph = bind_meaningful_prefixes(existing_graph)
    existing_graph.serialize(original_graph_path, format="turtle")
    logger.info(f"Saved original graph to {original_graph_path} with {len(existing_graph)} triples")
    
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
        logger.info(f"Using provided test_text (skipping API calls): '{test_text}'")
        # Set all text variables to test_text when using the predefined test text
        cleaned_text = test_text
        combined_text = test_text
        actual_verbalization = test_text  # Set actual_verbalization to test_text
        logger.info(f"Set actual_verbalization to test_text: '{actual_verbalization}'")
    else:
        # First, log how many relations we're about to process
        logger.info(f"Found {len(schematic_relations)} schematic relations to process for verbalization")
        
        # Process each relation with appropriate model
        logger.info("test_text is empty, proceeding with API calls for verbalization")
        for i, relation_graph in enumerate(schematic_relations):
            logger.info(f"=========== PROCESSING RELATION {i} ===========")
            # Serialize the relation graph
            relation_ttl = serialize_graph_as_turtle(relation_graph)
            
            # Save relation graph as string for inspection
            relation_ttl_file = os.path.join(pipeline_dir, f"relation_{i}_turtle.ttl")
            with open(relation_ttl_file, "w") as f:
                f.write(relation_ttl)
            logger.info(f"Saved relation Turtle content to: {relation_ttl_file}")
            
            # Debug triples content
            logger.info(f"Relation {i} has {len(relation_graph)} triples")
            
            # Get verbalization based on provided arguments
            verbalization = ""
            # Only use online verbalization if EXPLICITLY enabled with use_online_verbalization flag
            if args.use_online_verbalization:
                logger.info(f"VERBALIZATION METHOD: Using online model: {args.verbalization_model}")
                logger.info(f"API key provided: {'Yes' if args.verbalization_api_key else 'No'}")  # Don't log the actual key
                logger.info(f"API key first 4 chars: {args.verbalization_api_key[:4] if args.verbalization_api_key else 'None'}")
                
                # Check if anthropic package is available
                try:
                    import anthropic
                    logger.info("Anthropic package is available")
                except ImportError:
                    logger.error("ERROR: Anthropic package is NOT installed. Please run: pip install anthropic")
                    logger.error("Falling back to Ollama for verbalization")
                    verbalization = get_verbalization_from_ollama(
                        relation_ttl, 
                        model=args.model, 
                        output_dir=pipeline_dir
                    )
                else:
                    # If import successful, call Anthropic
                    verbalization = get_verbalization_from_anthropic(
                        relation_ttl, 
                        model=args.verbalization_model,
                        api_key=args.verbalization_api_key,
                        output_dir=pipeline_dir
                    )
            else:
                logger.info(f"VERBALIZATION METHOD: Using Ollama model: {args.model}")
                # Explicitly mention we're NOT using verbalization_model even if it's provided
                if args.verbalization_model:
                    logger.info(f"Verbalization model ({args.verbalization_model}) is provided but NOT being used")
                    logger.info(f"Using Ollama model ({args.model}) for verbalization instead")
                verbalization = get_verbalization_from_ollama(
                    relation_ttl, 
                    model=args.model, 
                    output_dir=pipeline_dir
                )
            
            # Add this debug line to see the actual verbalization
            logger.info(f"VERBALIZATION RESULT: '{verbalization}'")
            
            # Print verbalization to standard output for more visibility
            print(f"\n=========== VERBALIZATION FOR RELATION {i} ===========")
            print(f"{verbalization}")
            print("=========================================")
            
            # Save relation verbalization to a dedicated file
            relation_verbalization_file = os.path.join(pipeline_dir, f"relation_{i}_verbalization.txt")
            with open(relation_verbalization_file, "w") as f:
                f.write(verbalization)
            
            # Save to list only if we have a meaningful verbalization
            if verbalization and not verbalization.isspace():
                all_verbalizations.append(verbalization)
                logger.info(f"Added verbalization to list: '{verbalization}'")
            else:
                logger.warning(f"Empty verbalization for relation {i}, not adding to list")
            
            # Save the relation graph for reference
            relation_path = os.path.join(pipeline_dir, f"relation_{i}.ttl")
            relation_graph.serialize(relation_path, format="turtle")
            logger.info(f"Saved relation {i} to {relation_path}")
        
        # Combine verbalizations
        logger.info(f"Collected {len(all_verbalizations)} verbalizations to combine")
        combined_text = " ".join(all_verbalizations)
        logger.info(f"=========== COMBINED TEXT FOR AMR PARSING ===========")
        logger.info(f"Combined text: '{combined_text}'")
        
        # Handle case where we have no valid verbalizations
        if not combined_text or combined_text.isspace():
            logger.warning("No valid verbalizations collected, using fallback text")
            combined_text = "The object is related to another object."
        
        # Clean up the combined text
        cleaned_text = re.sub(r'^\s*\w+:\w+\s+', '', combined_text)
        cleaned_text = re.sub(r'^answer\s+', '', cleaned_text, flags=re.IGNORECASE)
        # Remove any remaining "Verbalization:" prefixes
        cleaned_text = re.sub(r'^"?Verbalization"?:\s*', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'"', '', cleaned_text)  # Remove any quotes
        
        # Extra check to ensure we don't have invalid verbalization text
        if cleaned_text.lower() == "verbalization" or not cleaned_text or cleaned_text.isspace():
            logger.warning(f"Invalid cleaned text detected: '{cleaned_text}', using fallback")
            # Use a more specific description based on what we know about the relations
            if "Occlusion" in str(existing_graph) and "knife" in str(existing_graph) and "apple" in str(existing_graph):
                cleaned_text = "The apple is occluding the knife."
            elif "Penetration" in str(existing_graph) and "knife" in str(existing_graph) and "apple" in str(existing_graph):
                cleaned_text = "The knife is penetrating the apple."
            else:
                cleaned_text = "The objects are interacting with each other."
        
        # Create a separate version of cleaned_text for storing the ACTUAL verbalization
        actual_verbalization = cleaned_text
        
    # Log final verbalization that will be used
    logger.info(f"Original text: '{combined_text}'")
    logger.info(f"Cleaned text for AMR parsing: '{cleaned_text}'")
    logger.info(f"Actual verbalization that will be used for alignment: '{actual_verbalization}'")
    
    # Save the text to be used for AMR parsing
    combined_text_file = os.path.join(pipeline_dir, "combined_text.txt")
    with open(combined_text_file, "w") as f:
        f.write(cleaned_text)
    logger.info(f"Saved text for AMR parsing to {combined_text_file}")
    
    # Load SPRING model if requested
    spring_model = None
    if args.use_spring:
        spring_model = load_spring_model(spring_model_file)
    
    # Display the final text to be used for AMR parsing
    print("\n=========== TEXT FOR AMR PARSING ===========")
    print(cleaned_text)
    print("===========================================\n")
    
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
    if args.online_services:
        logger.info(f"ONLINE SERVICES ENABLED: Will use WordNet, Framester SPARQL and Wikidata online services")
        
    new_graph_path = os.path.join(pipeline_dir, "new_graph.ttl")
    
    try:
        # Save the newly generated graph
        new_graph_str = converter.translate(
            amr=amr_representation, 
            serialize=True,
            text=cleaned_text,  # Pass the text for WSD services
            post_processing=True,
            # Use online services if flag is provided
            alt_api=args.online_services  # This enables alternative API endpoints
        )
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
        new_graph_path = os.path.join(pipeline_dir, "new_graph.ttl")
        # Apply meaningful prefixes before serialization
        new_graph = bind_meaningful_prefixes(new_graph)
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
        
        # Now that we have both graphs, perform a simple merge (union)
        logger.info("Performing graph merging")
        if args.online_services:
            logger.info("Using online services for merging with Wikidata and other services")
            # First, do a simple merge as a base
            merged_graph = converter.simple_merge_graphs(
                graph1=existing_graph,  # The graph loaded from CLI input
                graph2=new_graph       # The newly generated graph
            )
            # Then perform enhanced disambiguation and linking on the merged graph
            namespace = "http://example.org/log#"
            merged_graph = converter.translate_existing_graph(merged_graph, namespace, online_services=args.online_services)
            logger.info(f"Enhanced merged graph with online services, now has {len(merged_graph)} triples")
        else:
            # Regular simple merge (union) without online services
            merged_graph = converter.simple_merge_graphs(
                graph1=existing_graph,  # The graph loaded from CLI input
                graph2=new_graph       # The newly generated graph
            )
        
        # Save the merged graph
        merged_graph_path = os.path.join(pipeline_dir, "merged_graph.ttl")
        # Apply meaningful prefixes before serialization
        merged_graph = bind_meaningful_prefixes(merged_graph)
        merged_graph.serialize(merged_graph_path, format="turtle")
        logger.info(f"Saved merged graph to {merged_graph_path} with {len(merged_graph)} triples")

        # === Perform Entity Comparison/Alignment ===
        # Initialize the entity comparison class
        entity_comparator = EntityComparison()
        
        # Determine which entity alignment method to use based on CLI parameters
        if args.alignment_model and args.alignment_api_key:
            # Use LLM-based entity alignment with the actual verbalization
            logger.info("\n=========== PERFORMING LLM-BASED ENTITY ALIGNMENT ===========\n")
            # Use the actual verbalization instead of cleaned_text
            logger.info(f"Using actual verbalization for alignment: '{actual_verbalization}'")
            
            # Call the LLM alignment method
            alignment_results, added_triples = entity_comparator.llm_entity_alignment(
                original_graph=merged_graph,
                new_graph=new_graph,
                verbalization=actual_verbalization,
                api_key=args.alignment_api_key,
                model=args.alignment_model,
                output_dir=pipeline_dir
            )
            
            if "error" in alignment_results:
                logger.error(f"LLM alignment failed: {alignment_results.get('error')}")
                logger.warning("Falling back to similarity-based entity comparison")
                # Perform similarity-based entity comparison
                comparison_results, high_similarity_matches = entity_comparator.compare_graphs(existing_graph, new_graph)
                added_triples = entity_comparator.add_sameAs_triples_to_graph(merged_graph, comparison_results)
        else:
            # Perform standard similarity-based entity comparison (NLP-based)
            logger.info("\n=========== PERFORMING SIMILARITY-BASED ENTITY COMPARISON ===========\n")
            comparison_results, high_similarity_matches = entity_comparator.compare_graphs(existing_graph, new_graph)
            
            # Save comparison results to JSON
            comparison_file = os.path.join(pipeline_dir, "entity_comparison_results.json")
            try:
                with open(comparison_file, 'w') as f:
                    json.dump(comparison_results, f, indent=4)
                logger.info(f"Saved detailed comparison results to {comparison_file}")
            except Exception as e:
                logger.error(f"Error saving comparison results: {e}")
            
            # Add owl:sameAs triples to the merged graph using the entity comparator
            added_triples = entity_comparator.add_sameAs_triples_to_graph(merged_graph, comparison_results)
        
        logger.info("=========== ENTITY COMPARISON/ALIGNMENT COMPLETE ==========")
        # === End Entity Comparison/Alignment ===

        # Generate visualizations
        save_graph_visualizations(existing_graph, "original_graph", pipeline_dir)
        save_graph_visualizations(new_graph, "new_graph", pipeline_dir)
        save_graph_visualizations(merged_graph, "merged_graph", pipeline_dir)
        
        # If any triples were added, update the merged graph file and visualizations
        if added_triples > 0:
            logger.info(f"Resaving merged graph with {added_triples} new owl:sameAs relationships")
            # Apply meaningful prefixes before reserialization
            merged_graph = bind_meaningful_prefixes(merged_graph)
            # Save the updated merged graph
            merged_graph.serialize(merged_graph_path, format="turtle")
            # Regenerate visualization for the updated merged graph
            save_graph_visualizations(merged_graph, "merged_graph_with_sameAs", pipeline_dir)
        
        # Enrich the graph with Claude if requested - MOVED TO END OF PIPELINE
        if args.enrich:
            if args.verbalization_api_key:
                logger.info("=========== ENRICHING GRAPH WITH IMPLICIT KNOWLEDGE ===========")
                logger.info("Starting final enrichment step with Claude")
                print("\n==========================================")
                print("STARTING KNOWLEDGE GRAPH ENRICHMENT PROCESS")
                print("==========================================")
                
                # Record the start time
                enrichment_start_time = time.time()
                
                # Call the enrichment function
                enriched_graph = enrich_graph_with_claude(
                    graph=merged_graph,
                    text=cleaned_text,
                    api_key=args.verbalization_api_key,
                    output_dir=pipeline_dir
                )
                
                # Record the end time and calculate duration
                enrichment_end_time = time.time()
                enrichment_duration = enrichment_end_time - enrichment_start_time
                
                # Save visualization of the enriched graph
                save_graph_visualizations(enriched_graph, "enriched_graph", pipeline_dir)
                
                # Calculate the number of new triples added during enrichment
                new_triples_count = len(enriched_graph) - len(merged_graph)
                
                # Log and display completion message with statistics
                logger.info(f"Graph enrichment complete in {enrichment_duration:.2f} seconds")
                logger.info(f"Added {new_triples_count} new triples during enrichment")
                
                print("\n==========================================")
                print(f"KNOWLEDGE GRAPH ENRICHMENT COMPLETE")
                print(f"Time taken: {enrichment_duration:.2f} seconds")
                print(f"Original graph size: {len(merged_graph)} triples")
                print(f"Enriched graph size: {len(enriched_graph)} triples")
                print(f"New triples added: {new_triples_count}")
                print("==========================================")
                
                # Create a final pipeline summary file
                pipeline_summary_file = os.path.join(pipeline_dir, "pipeline_summary.txt")
                with open(pipeline_summary_file, "w") as f:
                    f.write("===== PIPELINE PROCESSING SUMMARY =====\n\n")
                    f.write(f"Input file: {input_graph_path}\n")
                    f.write(f"Processing completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    f.write("--- PIPELINE STEPS ---\n")
                    f.write(f"1. Loaded original graph: {len(existing_graph)} triples\n")
                    f.write(f"2. Generated verbalizations: {len(all_verbalizations)} for {len(schematic_relations)} relations\n")
                    f.write(f"3. AMR parsing with model: {spring_model_file}\n")
                    f.write(f"4. Generated new graph from AMR: {len(new_graph)} triples\n")
                    f.write(f"5. Merged graphs: {len(merged_graph)} triples\n")
                    f.write(f"6. Added entity alignments: {added_triples} owl:sameAs triples\n")
                    f.write(f"7. Enriched graph: {new_triples_count} new triples\n")
                    f.write(f"8. Final graph size: {len(enriched_graph)} triples\n\n")
                    
                    f.write("--- OUTPUT FILES ---\n")
                    f.write(f"Original graph: {original_graph_path}\n")
                    f.write(f"New graph from AMR: {new_graph_path}\n")
                    f.write(f"Merged graph: {merged_graph_path}\n")
                    f.write(f"Enriched graph: {os.path.join(pipeline_dir, 'enriched_graph.ttl')}\n")
                    f.write(f"Enrichment summary: {os.path.join(pipeline_dir, 'enrichment_summary.txt')}\n")
                    f.write(f"Verbalizations: {combined_text_file}\n\n")
                    
                    f.write("--- PROCESSING STATISTICS ---\n")
                    f.write(f"Enrichment time: {enrichment_duration:.2f} seconds\n")
                    f.write(f"New information ratio: {(new_triples_count / len(merged_graph)):.2%} increase\n")
                
                print(f"Pipeline summary saved to: {pipeline_summary_file}")
            else:
                error_msg = "Graph enrichment requested but no API key provided. Please provide --verbalization-api-key"
                logger.error(error_msg)
                print(f"\nERROR: {error_msg}")
        
        # Grounding enrichment
        if args.ground:
            logger.info("Starting grounding enrichment")
            enriched_graph = ground_enrichment(merged_graph, cleaned_text, output_dir=pipeline_dir)
            logger.info("Grounding enrichment completed")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        logger.exception("Detailed error information:")
        return


if __name__ == "__main__":
    main()
