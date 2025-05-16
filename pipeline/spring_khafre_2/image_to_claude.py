import anthropic
import argparse
import base64

parser = argparse.ArgumentParser(description="Process RDF graphs with AMR2FRED and Ollama using SPRING AMR parser")
parser.add_argument("--api-key", type=str,
                       help="API key for model")

args = parser.parse_args()

image_path = "/home/nick/spring_khafre_2/evt_4696812.9368284_cropped.jpg"
image_media_type = "image/jpeg"
with open(image_path, "rb") as image_file:
    image_data = base64.standard_b64encode(image_file.read()).decode("utf-8")

client = anthropic.Anthropic(api_key=args.api_key)

message = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=2048,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": "Describe this image."
                }
            ],
        }
    ],
)
print(message)
