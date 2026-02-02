import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not found in .env")
    exit(1)

genai.configure(api_key=api_key)

print("Listing available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error listing models: {e}")

print("\nTesting gemini-2.5-flash...")
try:
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content("Hello")
    print("Success!")
    print(response.text)
except Exception as e:
    print(f"Error with gemini-2.5-flash: {e}")

print("\nTesting gemini-1.5-flash...")
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Hello")
    print("Success!")
    print(response.text)
except Exception as e:
    print(f"Error with gemini-1.5-flash: {e}")
