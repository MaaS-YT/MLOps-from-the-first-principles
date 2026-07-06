"""
Test script for the monitoring system.

This script:
1. Sends 100 prediction requests to the API
2. Requests a drift report
3. Opens the generated HTML report in the browser
"""

import requests
import random
import time
import webbrowser
import os

API_BASE_URL = "http://localhost:8000"


def send_prediction_requests(num_requests=100):
    """Send prediction requests using a persistent session"""
    print(f"Sending {num_requests} prediction requests...")

    # Generate batch of data
    data_batch = [
        {
            "age": random.gauss(30, 4),
            "income": random.gauss(70000, 15000),
            "transactions": random.gauss(15, 5)
        }
        for _ in range(num_requests)
    ]

    start = time.time()
    session = requests.Session()

    try:
        # Send entire batch
        response = session.post(f"{API_BASE_URL}/predict/batch", json=data_batch)
        response.raise_for_status()
        result = response.json()
        print(f"  Batch processed: {result.get('count', 0)} predictions")

    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        return False
    finally:
        session.close()

    end = time.time()

    print(f"Successfully sent {num_requests} requests")
    print(
        f"Total time taken: {end - start:.4f} seconds ({num_requests / (end - start):.1f} req/s)")
    return True


def request_drift_report():
    """Request the drift report from the API"""
    print("\nRequesting drift report...")

    try:
        response = requests.get(f"{API_BASE_URL}/drift-report")
        response.raise_for_status()
        result = response.json()

        print("\nDrift Report Result:")
        print(f"  Message: {result.get('message')}")
        print(f"  Dataset Drift: {result.get('dataset_drift')}")
        print(f"  Report Path: {result.get('report_path')}")
        print(f"  Total Requests: {result.get('total_requests', 'N/A')}")

        return result

    except requests.exceptions.RequestException as e:
        print(f"Error requesting drift report: {e}")
        return None


def open_drift_report(report_path="drift_report.html"):
    """Open the drift report HTML file in the default browser"""
    if os.path.exists(report_path):
        print(f"\nOpening {report_path} in browser...")
        webbrowser.open(f"file://{os.path.abspath(report_path)}")
        print("Report opened in browser")
    else:
        print(f"Warning: {report_path} not found")


def check_server_health():
    """Check if the server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/docs", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def main():
    print("=" * 60)
    print("Monitoring System Test Script")
    print("=" * 60)

    # Check if server is running
    print("\nChecking server status...")
    if not check_server_health():
        print("ERROR: Server is not running!")
        print("\nPlease start the server first:")
        print("  uvicorn app:app --reload --port 8000")
        return

    print("Server is running")

    # Send prediction requests
    if not send_prediction_requests(100):
        print("ERROR: Failed to send requests")
        return

    # Wait a moment for logs to be written
    time.sleep(1)

    # Request drift report
    result = request_drift_report()
    if not result:
        print("ERROR: Failed to generate drift report")
        return

    # Open the report in browser
    report_path = result.get('report_path', 'drift_report.html')
    open_drift_report(report_path)

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
