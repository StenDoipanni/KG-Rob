# Hybrid Knowledge Graph Augmentation for Cognitive Agents with Common Sense
This repository presents code, additional materials, and evaluation for our framework for knowledge graph augmentation grounded in cognitive science and embodied cognition. </br>
Building on several existing methods, we propose a modular and hybrid neuro-symbolic architecture. Beginning with a neural component, we perform object recognition on an image, generate well-formed RDF knowledge graphs, and enrich these graphs using existing Semantic Web resources such as PropBank, WordNet, VerbAtlas, and alignments with the DOLCE foundational ontology. This includes the localization of RDF entities with respect to specific regions of the original image. Finally, we leverage the commonsense knowledge encoded in large language models (LLMs) to further enrich the graph with tacit and implicit information, relevant to particular use cases of cognitive agents, such as likely future outcomes, which can help identify potential dangerous actions, and can lead to risk reduction and danger avoidance interactions.


### Useful CLI commands


The complete list of CLI arguments is:
```
--input : Path to input graph file
--use-spring : Use SPRING parser instead of simple AMR generator
--spring-model : SPRING model file to use. The default one is AMR3.parsing.pt
--model : Ollama model to use for verbalization (default llama3.1:8b)
--verbalization-model : Online model to use for verbalization (e.g. claude-3-7-sonnet-20250219)
--verbalization-api-key : API key for verbalization model (required if using --verbalization-model)
--alignment-model : Model to use for entity alignment (e.g. claude-3-7-sonnet-20250219)
--enrich : Enable graph enrichment with implicit knowledge
--enrich-model : Model to use for graph enrichment ( default claude-3-7-sonnet-20250219)
--enrich-api-key : API key for enrichment model (required if using --enrich)
--ground : Enable graph grounding with classes and roles from the Khafre ontology 
```





**IMPORTANT** </br>
AMR3.parsing.pt model can be downloaded from here: [AMR3.generation-1.0.tar.bz2](http://nlp.uniroma1.it/AMR/AMR3.generation-1.0.tar.bz2) and should be located like this `./spring_khafre_2/spring_amr/AMR3.parsing.pt`

**IMPORTANT**
The file with mappings to wikidata can be found [here](http://hrilabdemo.ddns.net/index_enwiki-latest.zip), and should be located at `./spring_khafre_2/index_enwiki-latest.db`


Command to verbalize using Ollama locally:
```
python main.py --input evt_23492.5989675.ttl --model llama3.1:8b --use-spring --spring-model AMR3.parsing.pt --online-services
```

To run the following commands you will need to provide an Anthropic API key, instructions to create one can be found [here](https://docs.anthropic.com/en/docs/initial-setup).
You can then export your key value with:
```
export API_KEY="insert-your-api-key"
```

Command to verbalize using Claude 3.7, using as input the `chaplin.ttl`file.
```
python main.py --input chaplin.ttl --verbalization-model claude-3-7-sonnet-20250219 --verbalization-api-key $API_KEY --use-spring --spring-model AMR3.parsing.pt --online-services 
```


Verbalize AND Align using Claude 3.7
```
python main.py --input evt_23492.5989675.ttl --verbalization-model claude-3-7-sonnet-20250219 --verbalization-api-key $API_KEY --alignment-model claude-3-opus-20240229 --alignment-api-key $API_KEY --enrich --enrich-model claude-3-sonnet-20240229 --enrich-api-key $API_KEY --use-spring --spring-model AMR3.parsing.pt --online-services
```

Enrich with Claude 3.7
```
python main.py --input evt_23492.5989675.ttl --verbalization-model claude-3-7-sonnet-20250219 --verbalization-api-key $API_KEY --alignment-model claude-3-opus-20240229 --alignment-api-key $API_KEY --enrich --enrich-model claude-3-sonnet-20240229 --enrich-api-key $API_KEY --use-spring --spring-model AMR3.parsing.pt --online-services
```

Ground using Claude 3.7 and Classes and Roles from the [Khafre](https://github.com/Heideggerian-AI-v5/khafre/blob/main/axioms/khafre.ttl) ontology
```
python main.py --input evt_23492.5989675.ttl --verbalization-model claude-3-7-sonnet-20250219 --verbalization-api-key $API_KEY --alignment-model claude-3-opus-20240229 --alignment-api-key $API_KEY --enrich --enrich-model claude-3-sonnet-20240229 --enrich-api-key $API_KEY --use-spring --spring-model AMR3.parsing.pt --online-services
```

