import requests

def test_origin(origin, expected_status=200):
    url = "http://127.0.0.1:8001/api/session"
    headers = {
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    }
    
    print(f"Testing Origin: {origin}")
    
    # Preflight Request
    try:
        response = requests.options(url, headers=headers)
        print(f"  Preflight: Status {response.status_code}")
        if "Access-Control-Allow-Origin" in response.headers:
            print(f"  Preflight: Allow-Origin: {response.headers['Access-Control-Allow-Origin']}")
        else:
            print("  Preflight: Allow-Origin NOT found in headers")
    except Exception as e:
        print(f"  Preflight: Error: {e}")

    # Actual Request
    try:
        response = requests.post(url, json={"index_id": 1}, headers={"Origin": origin})
        print(f"  Actual: Status {response.status_code}")
        if "Access-Control-Allow-Origin" in response.headers:
            print(f"  Actual: Allow-Origin: {response.headers['Access-Control-Allow-Origin']}")
        else:
            print("  Actual: Allow-Origin NOT found in headers")
    except Exception as e:
        print(f"  Actual: Error: {e}")
    print("-" * 30)

if __name__ == "__main__":
    # Test allowed origin
    test_origin("http://localhost:8081")
    
    # Test another allowed origin
    test_origin("http://127.0.0.1:8081")
    
    # Test disallowed origin
    test_origin("http://evil.com")
