import requests
import os

BASE_URL = "http://localhost:8000"
INDEX_ID = 888

def test_append():
    # 1. Create test files
    with open("test1.txt", "w") as f:
        f.write("This is the first part of the document.")
    
    with open("test2.txt", "w") as f:
        f.write("This is the second part of the document.")

    try:
        # 2. Upload first file
        print(f"Uploading test1.txt for Index {INDEX_ID}...")
        with open("test1.txt", "rb") as f:
            files = {"file": ("test1.txt", f, "text/plain")}
            data = {"index_id": INDEX_ID}
            response = requests.post(f"{BASE_URL}/api/upload", files=files, data=data)
            print(f"Response: {response.json()}")

        # 3. Upload second file
        print(f"Uploading test2.txt for Index {INDEX_ID}...")
        with open("test2.txt", "rb") as f:
            files = {"file": ("test2.txt", f, "text/plain")}
            data = {"index_id": INDEX_ID}
            response = requests.post(f"{BASE_URL}/api/upload", files=files, data=data)
            print(f"Response: {response.json()}")

        # 4. Verify combined content
        print(f"Verifying content for Index {INDEX_ID}...")
        response = requests.post(f"{BASE_URL}/api/session", json={"index_id": INDEX_ID})
        result = response.json()
        content = result.get("content", "")
        print(f"Combined content: {content}")

        if "first part" in content and "second part" in content:
            print("SUCCESS: Data was appended correctly!")
        else:
            print("FAILURE: Data was not appended correctly.")

    finally:
        # Clean up
        if os.path.exists("test1.txt"):
            os.remove("test1.txt")
        if os.path.exists("test2.txt"):
            os.remove("test2.txt")

if __name__ == "__main__":
    test_append()
