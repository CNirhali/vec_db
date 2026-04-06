import requests
import sys

BASE_URL = "http://localhost:8000"
API_KEY = "supersecretkey"

def test_negative_content_length():
    print("Testing negative Content-Length...")
    # Note: Modern servers like Uvicorn/h11 often reject negative Content-Length before it reaches middleware.
    # We still want to see if we get a 400.
    import http.client
    url = "localhost"
    port = 8000
    conn = http.client.HTTPConnection(url, port)
    headers = {
        "X-API-Key": API_KEY,
        "Content-Length": "-1"
    }
    try:
        conn.request("POST", "/status", body="", headers=headers)
        response = conn.getresponse()
        print(f"Status: {response.status}")
        data = response.read().decode()
        print(f"Response: {data}")
        # Either our middleware or Uvicorn should return 400
        assert response.status == 400
    finally:
        conn.close()

def test_malformed_content_length():
    print("Testing malformed Content-Length...")
    import http.client
    url = "localhost"
    port = 8000
    conn = http.client.HTTPConnection(url, port)
    headers = {
        "X-API-Key": API_KEY,
        "Content-Length": "abc"
    }
    try:
        conn.request("POST", "/status", body="", headers=headers)
        response = conn.getresponse()
        print(f"Status: {response.status}")
        data = response.read().decode()
        print(f"Response: {data}")
        assert response.status == 400
        # Our middleware handles ValueError
        if "Invalid Content-Length" in data:
             print("Middleware caught malformed Content-Length")
    finally:
        conn.close()

def test_enhanced_security_headers():
    print("Testing enhanced security headers...")
    response = requests.get(f"{BASE_URL}/status", headers={"X-API-Key": API_KEY})

    expected_headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none';",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
        "X-XSS-Protection": "0",
        "Referrer-Policy": "no-referrer",
        "X-Permitted-Cross-Domain-Policies": "none",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Permissions-Policy": "accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()"
    }

    for header, expected_value in expected_headers.items():
        assert header in response.headers, f"Missing header: {header}"
        assert response.headers[header] == expected_value, f"Incorrect value for {header}: expected '{expected_value}' but got '{response.headers[header]}'"

    print("All enhanced security headers are present and correct.")

if __name__ == "__main__":
    try:
        test_negative_content_length()
        test_malformed_content_length()
        test_enhanced_security_headers()
        print("NEW SECURITY PROTECTIONS TESTS PASSED")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)
