#!/bin/bash
set -e

run_test() {
    echo "Running $1..."
    # Kill any existing server
    kill $(pgrep -f "run_api.py") 2>/dev/null || true
    sleep 2
    # Clear data
    rm -f *.h5 *.bin *.log
    # Start server
    python run_api.py > server.log 2>&1 &
    # Wait for server to start
    sleep 5
    # Run test
    python "$1"
    echo "$1 PASSED"
    echo "-----------------------------------"
}

run_test test_headers.py
run_test test_security.py
run_test test_concurrency.py
run_test test_filter.py
run_test test_rate_limiting.py
run_test test_request_limit.py

echo "ALL TEST FILES PASSED"
