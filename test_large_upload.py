import requests
import os

def test_large_upload():
    url = "http://127.0.0.1:8001/api/bulk-upload"
    
    # Generate a large file (~85KB)
    content = "This is a test line. " * 4200 
    filename = "997.txt"
    password = "admin123" # Default from .env edit
    
    with open(filename, "w") as f:
        f.write(content)
        
    print(f"File size: {os.path.getsize(filename)} bytes")
    
    try:
        files = [("files", (filename, open(filename, "rb")))]
        data = {"password": password}
        
        print(f"Sending request to {url}...")
        response = requests.post(url, files=files, data=data)
        
        print("Response status code:", response.status_code)
        print("Response body:", response.json())
        
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results and results[0]["status"] == "success":
                print("SUCCESS: Large file uploaded and truncated correctly.")
            else:
                print("FAILURE: Upload did not return success.")
        else:
            print(f"FAILURE: Status code {response.status_code}")
            
    finally:
        # Cleanup
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    test_large_upload()
