import requests
import sys

BASE_URL = "http://localhost:8000"
API_KEY = "supersecretkey"

def test_security_headers_on_413():
    # Send a large body to trigger 413
    # Use a stream to avoid memory issues in the test script
    def large_gen():
        yield "x" * (150 * 1024 * 1024 + 1024)

    response = requests.post(
        f"{BASE_URL}/status",
        data=large_gen(),
        headers={"X-API-Key": API_KEY, "Content-Length": str(150 * 1024 * 1024 + 1024)}
    )

    print(f"Status: {response.status_code}")
    assert response.status_code == 413, f"Expected 413 but got {response.status_code}"

    headers_to_check = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Content-Security-Policy": "default-src 'self'; script-src 'self'; object-src 'none';",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "X-Permitted-Cross-Domain-Policies": "none"
    }

    for header, expected_value in headers_to_check.items():
        assert header in response.headers, f"Missing header: {header}"
        assert response.headers[header] == expected_value, f"Incorrect value for {header}: expected '{expected_value}' but got '{response.headers[header]}'"

    print("All security headers present and correct on 413 response.")

def test_security_headers_on_405():
    # Trigger 405 Method Not Allowed
    response = requests.post(
        f"{BASE_URL}/status",
        headers={"X-API-Key": API_KEY}
    )

    print(f"Status: {response.status_code}")
    assert response.status_code == 405, f"Expected 405 but got {response.status_code}"

    headers_to_check = [
        "X-Content-Type-Options",
        "X-Frame-Options",
        "Content-Security-Policy",
        "Strict-Transport-Security",
        "X-XSS-Protection",
        "Referrer-Policy",
        "X-Permitted-Cross-Domain-Policies"
    ]

    for header in headers_to_check:
        assert header in response.headers, f"Missing header: {header}"

    print("All security headers present on 405 response.")

if __name__ == "__main__":
    try:
        test_security_headers_on_413()
        test_security_headers_on_405()
        print("HEADER SECURITY TESTS PASSED")
    except AssertionError as e:
        print(f"HEADER SECURITY TESTS FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
