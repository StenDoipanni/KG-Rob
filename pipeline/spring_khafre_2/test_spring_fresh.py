import sys
import os
import importlib

# Get the absolute path to the spring directory
current_dir = os.path.dirname(os.path.abspath(__file__))
spring_amr_dir = os.path.abspath(os.path.join(current_dir, 'spring_amr'))

# Add the spring_amr directory to the Python path
if spring_amr_dir not in sys.path:
    sys.path.insert(0, spring_amr_dir)

print(f"Current script location: {os.path.abspath(__file__)}")
print(f"Current directory: {current_dir}")
print(f"Spring AMR directory: {spring_amr_dir}")
print(f"Python path: {sys.path}")

# Check if spring_amr files exist
print("\nChecking for required files:")
required_files = ['tokenization_bart.py', 'modeling_bart.py', '__init__.py']
for file in required_files:
    file_path = os.path.join(spring_amr_dir, file)
    print(f"{file}: {'Exists' if os.path.exists(file_path) else 'Missing'}")

try:
    # Import the modules in a specific order to avoid circular imports
    import spring_amr.penman
    import spring_amr.tokenization_bart
    import spring_amr.modeling_bart
    
    from spring_amr.tokenization_bart import AMRBartTokenizer
    from spring_amr.modeling_bart import AMRBartForConditionalGeneration
    
    print("\nImports successful!")
    print(f"AMRBartTokenizer location: {AMRBartTokenizer.__module__}")
    print(f"AMRBartForConditionalGeneration location: {AMRBartForConditionalGeneration.__module__}")
except ImportError as e:
    print(f"\nImport error: {str(e)}")
    print("\nAvailable modules in spring_amr:")
    try:
        import spring_amr
        print(f"spring_amr location: {spring_amr.__file__}")
        print(f"spring_amr contents: {dir(spring_amr)}")
    except ImportError:
        print("Could not import spring_amr at all")

def test_spring_model():
    try:
        # Initialize tokenizer
        tokenizer = AMRBartTokenizer.from_pretrained('facebook/bart-large')
        print("\nTokenizer loaded successfully")
        
        # Initialize model
        model = AMRBartForConditionalGeneration.from_pretrained('facebook/bart-large')
        print("Model loaded successfully")
        
        return True
    except Exception as e:
        print(f"Error loading SPRING model: {str(e)}")
        return False

if __name__ == "__main__":
    test_spring_model()
