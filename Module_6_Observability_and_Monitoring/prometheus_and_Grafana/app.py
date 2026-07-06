import pandas as pd
from evidently.presets import DataDriftPreset
from evidently import Report
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, Gauge
from starlette_exporter import PrometheusMiddleware, handle_metrics
from pydantic import BaseModel
from typing import List
import logging
import os
import csv
from psutil import cpu_percent, virtual_memory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Persistent log file
LOG_FILE = "request_log.csv"

# Prometheus metrics
REQUEST_COUNT = Counter("prediction_requests", "Total prediction requests")
LATENCY = Histogram("prediction_latency_seconds", "Model latency")
CPU_USAGE = Gauge("system_cpu_usage", "System CPU usage percentage")
MEMORY_USAGE = Gauge("system_memory_usage", "System memory usage percentage")

app = FastAPI()

# Add Prometheus middleware
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)


def load_request_log():
    """Load existing request log from CSV or create new DataFrame"""
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            logger.info(f"Loaded {len(df)} existing requests from {LOG_FILE}")
            return df
        except Exception as e:
            logger.error(f"Error loading log file: {e}")
            return pd.DataFrame(columns=["age", "income", "transactions"])
    return pd.DataFrame(columns=["age", "income", "transactions"])


def append_request_log(data_dict):
    """Append a single request to CSV file"""
    try:
        file_exists = os.path.exists(LOG_FILE)
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=["age", "income", "transactions"])
            if not file_exists:
                writer.writeheader()
            writer.writerow(data_dict)
    except Exception as e:
        logger.error(f"Error appending to log file: {e}")


def append_batch_request_log(data_list):
    """Append multiple requests to CSV file"""
    try:
        file_exists = os.path.exists(LOG_FILE)
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=["age", "income", "transactions"])
            if not file_exists:
                writer.writeheader()
            writer.writerows(data_list)
        logger.info(f"Appended {len(data_list)} requests to {LOG_FILE}")
    except Exception as e:
        logger.error(f"Error appending batch to log file: {e}")


class InputData(BaseModel):
    age: float
    income: float
    transactions: float


@app.post("/predict")
def predict(data: InputData):
    REQUEST_COUNT.inc()

    with LATENCY.time():
        # dummy model
        pred = (0.1 * data.age) + (0.0002 * data.income) + \
            (0.5 * data.transactions)

    # Log the request for drift detection (append only - much faster)
    append_request_log(data.dict())

    return {"prediction": pred}


@app.post("/predict/batch")
def predict_batch(data_list: List[InputData]):
    """Handle batch predictions efficiently"""
    REQUEST_COUNT.inc(len(data_list))

    predictions = []
    data_dicts = []

    for data in data_list:
        with LATENCY.time():
            # dummy model
            pred = (0.1 * data.age) + (0.0002 * data.income) + \
                (0.5 * data.transactions)
        predictions.append({"prediction": pred})
        data_dicts.append(data.dict())

    # Log all requests in one operation
    append_batch_request_log(data_dicts)
    logger.info(f"Processed batch of {len(data_list)} predictions")

    return {"predictions": predictions, "count": len(predictions)}


@app.get("/drift-report")
def drift_report():

    df_curr = load_request_log()

    if len(df_curr) < 50:
        logger.warning(
            f"Insufficient data for drift report: {len(df_curr)} requests")
        return {"error": "Not enough data for drift report", "current_count": len(df_curr)}

    # Load reference data (training)
    df_ref = pd.read_csv("reference.csv")

    report = Report(metrics=[DataDriftPreset()])
    report = report.run(reference_data=df_ref, current_data=df_curr)

    report.save_html("drift_report.html")

    # Extract drift information from the new Evidently structure
    report_dict = report.dict()

    # Get drifted columns count
    drifted_columns = report_dict["metrics"][0]["value"]["count"]
    drift_share = report_dict["metrics"][0]["value"]["share"]

    # Check individual column drift (True means drift detected, p-value < threshold)
    column_drifts = {}
    # Skip first metric (DriftedColumnsCount)
    for metric in report_dict["metrics"][1:]:
        if "column" in metric["config"]:
            column = metric["config"]["column"]
            p_value = metric["value"]
            threshold = metric["config"]["threshold"]
            drift_detected = p_value < threshold
            column_drifts[column] = {
                "p_value": float(p_value),
                "drift_detected": bool(drift_detected)
            }

    dataset_drift = bool(drifted_columns > 0)

    logger.info(
        f"Drift report generated with {len(df_curr)} requests. Drift detected: {dataset_drift}, Drifted columns: {drifted_columns}")

    return {
        "message": "Report generated",
        "dataset_drift": dataset_drift,
        "drifted_columns_count": int(drifted_columns),
        "drift_share": float(drift_share),
        "column_drifts": column_drifts,
        "report_path": "drift_report.html",
        "total_requests": len(df_curr)
    }


@app.on_event("startup")
def start_monitoring():
    """Start monitoring system metrics."""
    import threading
    import time

    def monitor_system_metrics():
        while True:
            CPU_USAGE.set(cpu_percent(interval=1))
            MEMORY_USAGE.set(virtual_memory().percent)
            time.sleep(1)  # Adjust the frequency of updates as needed

    thread = threading.Thread(target=monitor_system_metrics, daemon=True)
    thread.start()
