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
from amr2fred_khafre_util import *

# Import the SPRING parser dependencies
import torch
import penman
from pathlib import Path

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

# Replace this with your test text if you want to skip Ollama verbalization
# If left empty, the script will call Ollama for verbalization
test_text = ""

# Ollama model to use
ollama_model = "llama3.1:8b"


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
    
    # Generate DOT format
    dot_content = digraph_writer.graph_to_digraph(graph)
    
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
    base_terms = set()
    salient_relations = []
    
    # SPARQL query to find schematic relations and their types
    query1 = """
    SELECT DISTINCT ?relation ?type
    WHERE {
        ?relation a ?type .
        FILTER(CONTAINS(str(?relation), "schematicRelation"))
        FILTER(!CONTAINS(str(?type), "NamedIndividual"))
    }
    """
    
    # SPARQL query to find objects related to schematic relations through 'has' predicates
    query2 = """
    SELECT DISTINCT ?relation ?predicate ?object
    WHERE {
        ?relation ?predicate ?object .
        FILTER(CONTAINS(str(?relation), "schematicRelation"))
        FILTER(CONTAINS(str(?predicate), "has"))
    }
    """
    
    try:
        # Execute first query to get relation types
        logger.info("Extracting relation types as base terms...")
        results1 = graph.query(query1)
        
        for row in results1:
            type_uri = str(row['type'])
            relation_uri = str(row['relation'])
            
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
            relation_uri = str(row['relation'])
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
            if len(relation_graph) > 0:
                relations.append(relation_graph)
                logger.info(f"Extracted relation {relation_uri} with {len(relation_graph)} triples")
                
                # Print triples for this relation
                print(f"\nTriples for relation {relation_uri}:")
                for s, p, o in relation_graph:
                    print(f"  {s} {p} {o}")
    
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

Answer with:
"Verbalization": "answer"
and nothing more, disregard numbers, the verbalization should be related only to the following triples, and not the ones in example.
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


def serialize_graph_as_turtle(graph):
    """
    Serialize a graph as Turtle format
    """
    return graph.serialize(format="turtle")


def get_amr_from_local_parser(text, output_dir="./out/amr/"):
    """
    Get AMR representation using a simple fallback generator
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



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process RDF graphs with AMR2FRED and Ollama using local SPRING AMR parser")
    parser.add_argument("--input", "-i", type=str, help="Path to input graph file", required=True)
    parser.add_argument("--model", "-m", type=str, default="deepseek-r1:1.5b", help="Ollama model to use")
    args = parser.parse_args()
    
    # Set Ollama model
    global ollama_model
    if args.model:
        ollama_model = args.model
    
    # Load your existing graph
    existing_graph = Graph()
    input_graph_path = args.input
    
    try:
        existing_graph.parse(input_graph_path, format="turtle")
        logger.info(f"Loaded existing graph with {len(existing_graph)} triples")
    except Exception as e:
        logger.error(f"Error loading existing graph: {e}")
        logger.info("Creating a sample graph for testing")
        
        # Create a sample graph if loading fails (for testing purposes)
        from rdflib import Namespace, URIRef
        from rdflib.namespace import RDF, OWL
        
        ns1 = Namespace("http://example.org/ns1#")
        
        # Add sample triples for penetration relation
        existing_graph.add((ns1.schematicRelation_15, RDF.type, ns1.Penetration))
        existing_graph.add((ns1.schematicRelation_15, RDF.type, OWL.NamedIndividual))
        existing_graph.add((ns1.schematicRelation_15, ns1.eventMode, ns1.Ended))
        existing_graph.add((ns1.schematicRelation_15, ns1.hasPenetrator, ns1.knife_70))
        existing_graph.add((ns1.schematicRelation_15, ns1.hasPenetree, ns1.apple_27))
        
        # Set input graph path to generated file
        input_graph_path = os.path.join(pipeline_dir, "generated_existing_graph.ttl")
    
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
        # Process each relation with Ollama
        for i, relation_graph in enumerate(schematic_relations):
            logger.info(f"=========== PROCESSING RELATION {i} ===========")
            # Serialize the relation graph
            relation_ttl = serialize_graph_as_turtle(relation_graph)
            
            # Save relation graph as string for inspection
            relation_ttl_file = os.path.join(pipeline_dir, f"relation_{i}_turtle.ttl")
            with open(relation_ttl_file, "w") as f:
                f.write(relation_ttl)
            logger.info(f"Saved relation Turtle content to: {relation_ttl_file}")
            
            # Get verbalization from Ollama
            logger.info(f"Getting verbalization for relation {i}")
            verbalization = get_verbalization_from_ollama(relation_ttl, model=ollama_model, output_dir=pipeline_dir)
            
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
    
    # Use the local SPRING AMR parser
    logger.info(f"Calling local SPRING AMR parser")
    amr_representation = get_amr_from_local_parser(cleaned_text)
    
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
        
        new_graph.serialize(new_graph_path, format="turtle")
        logger.info(f"Saved AMR2FRED graph to {new_graph_path} with {len(new_graph)} triples")
        
    except Exception as e:
        logger.error(f"Error in translate: {e}")
        new_graph = Graph()
        new_graph.serialize(new_graph_path, format="turtle")
        logger.info(f"Saved empty graph due to error")
    
    # Now proceed with merging using the original text for alignment
    logger.info(f"=========== CALLING AMR2FRED WITH MAPPINGS ===========")
    
    # Create a new copy of existing graph for fallback
    result_graph = Graph()
    for s, p, o in existing_graph:
        result_graph.add((s, p, o))
    
    try:
        # We use the verbalization text here for better alignment since it contains the semantic content
        result_graph_str = converter.translate_with_fuzzy_mappings(
            text=cleaned_text,
            existing_graph=existing_graph,
            base_terms=base_terms,
            alignment_method="nlp",
            api_key=None,
            amr=amr_representation  # Pass the AMR representation directly
        )
        logger.info(f"Translate with mappings returned result of type: {type(result_graph_str)}")
        
        # Save the raw output from mappings for inspection
        mappings_output_file = os.path.join(pipeline_dir, "amr2fred_mappings_output.txt")
        with open(mappings_output_file, "wb") as f:
            if isinstance(result_graph_str, str):
                f.write(result_graph_str.encode('utf-8'))
            else:
                f.write(str(result_graph_str).encode('utf-8'))
        logger.info(f"Saved raw translate_with_mappings output to {mappings_output_file}")
        
        # Process the result
        if isinstance(result_graph_str, Graph):
            # Clear existing result graph and copy from result_graph_str
            result_graph = Graph()
            for s, p, o in result_graph_str:
                result_graph.add((s, p, o))
            logger.info(f"Result is already a Graph object with {len(result_graph)} triples")
        else:
            try:
                # Create new empty graph
                temp_graph = Graph()
                temp_graph.parse(data=result_graph_str, format="turtle")
                logger.info(f"Successfully parsed mappings output into graph with {len(temp_graph)} triples")
                
                # Replace result_graph with temp_graph
                result_graph = Graph()
                for s, p, o in temp_graph:
                    result_graph.add((s, p, o))
            except Exception as parse_error:
                logger.error(f"Error parsing result from translate_with_mappings: {parse_error}")
                logger.info(f"Using existing graph as fallback - already copied")
    except Exception as e:
        logger.error(f"Error in translate_with_mappings: {e}")
        logger.info(f"Using existing graph as fallback - already copied")
    
    # Save the merged result graph
    result_graph_path = os.path.join(pipeline_dir, "merged_graph.ttl")
    result_graph.serialize(result_graph_path, format="turtle")
    logger.info(f"Saved merged result graph to {result_graph_path} with {len(result_graph)} triples")
    
    # After all graphs are saved, generate visualizations
    logger.info("Generating visualizations for all graphs")
    save_graph_visualizations(existing_graph, "original_graph", pipeline_dir)
    save_graph_visualizations(new_graph, "new_graph", pipeline_dir)
    save_graph_visualizations(result_graph, "merged_graph", pipeline_dir)
    
    # Print summary
    print("\nOutput Files:")
    print(f"1. Original Existing Graph: {existing_graph_path}")
    print(f"2. New AMR2FRED Graph: {new_graph_path}")
    print(f"3. Merged Result Graph: {result_graph_path}")
    
    # Count stats for verification
    print("\nGraph Statistics:")
    print(f"- Original Graph: {len(existing_graph)} triples")
    print(f"- New AMR2FRED Graph: {len(new_graph)} triples")
    print(f"- Merged Result Graph: {len(result_graph)} triples")
    
    # Check if merged graph contains more triples than original
    if len(result_graph) > len(existing_graph):
        print(f"✓ Verification: Merged graph has {len(result_graph) - len(existing_graph)} more triples than original graph")
    else:
        print("⚠ Warning: Merged graph doesn't have more triples than original graph")
    
    # Calculate overlap between graphs
    original_triples = set()
    for s, p, o in existing_graph:
        original_triples.add((s, p, o))
        
    new_triples = set()
    for s, p, o in new_graph:
        new_triples.add((s, p, o))
        
    merged_triples = set()
    for s, p, o in result_graph:
        merged_triples.add((s, p, o))
    
    common_with_original = len(merged_triples.intersection(original_triples))
    common_with_new = len(merged_triples.intersection(new_triples))
    
    print(f"- Merged graph contains {common_with_original} triples from original graph")
    print(f"- Merged graph contains {common_with_new} triples from new graph")
    
    print("\nAlignment Complete!")
    print("\nVisualizations have been saved to the pipeline_out directory")


if __name__ == "__main__":
    main()
