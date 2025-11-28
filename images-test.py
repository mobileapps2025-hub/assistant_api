import os
import sys
import openai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY environment variable not set.")
    sys.exit(1)

client = openai.OpenAI(api_key=api_key)

def analyze_image(image_url):
    try:
        response = client.chat.completions.create(
            model="gpt-5.1", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What the work order trend graphic is telling me about?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            max_completion_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

if __name__ == "__main__":
    default_url = "https://cdn.dribbble.com/userupload/44400117/file/ae908f105ab96a1f7782adfbe148ea0e.jpg?resize=1024x984&vertical=center"
    
    url = sys.argv[1] if len(sys.argv) > 1 else default_url
    
    print(f"Analyzing image: {url}")
    result = analyze_image(url)
    print("\nResult:")
    print(result)
