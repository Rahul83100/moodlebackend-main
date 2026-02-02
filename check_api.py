import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key found: {api_key[:10]}..." if api_key else "API Key NOT found")

if not api_key:
    exit(1)

genai.configure(api_key=api_key)

print("Testing gemini-2.5-flash...")
try:
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content("Hello, this is a test.")
    print("Success! Response:")
    print(response.text)
except Exception as e:
    print(f"Error: {e}")
