"""
Grafana Dashboard Generator for FastAPI Monitoring Application

This script generates a comprehensive Grafana dashboard for monitoring:
- Request counts and latency
- System metrics (CPU, memory)
- Data drift detection from Evidently
"""

from grafanalib.core import (
    Dashboard, TimeSeries, Target, GridPos,
    RowPanel, Templating, Time
)
from grafanalib._gen import DashboardEncoder
import json


def generate_dashboard():
    """Generate a comprehensive monitoring dashboard"""

    dashboard = Dashboard(
        title="FastAPI Monitoring - Prediction Service",
        description="Monitoring dashboard for ML prediction service with Prometheus metrics",
        tags=["fastapi", "ml", "monitoring", "prometheus"],
        timezone="browser",
        refresh="10s",
        time=Time("now-1h", "now"),
        panels=[
            # Row 1: Request Metrics
            RowPanel(gridPos=GridPos(h=1, w=24, x=0, y=0),
                     title="Request Metrics"),

            # Total Requests Counter
            TimeSeries(
                title="Total Prediction Requests",
                dataSource="prometheus",
                targets=[
                    Target(
                        expr='prediction_requests_total',
                        legendFormat='Total Requests',
                        refId='A',
                    ),
                ],
                gridPos=GridPos(h=8, w=8, x=0, y=1),
            ),

            # Request Rate
            TimeSeries(
                title="Request Rate (req/s)",
                dataSource="prometheus",
                targets=[
                    Target(
                        expr='rate(prediction_requests_total[1m])',
                        legendFormat='Requests/sec',
                        refId='A',
                    ),
                ],
                gridPos=GridPos(h=8, w=8, x=8, y=1),
            ),

            # HTTP Status Codes
            TimeSeries(
                title="HTTP Status Codes",
                dataSource="prometheus",
                targets=[
                    Target(
                        expr='rate(starlette_requests_total[1m])',
                        legendFormat='{{status}} - {{method}} {{path_template}}',
                        refId='A',
                    ),
                ],
                gridPos=GridPos(h=8, w=8, x=16, y=1),
            ),

            # Row 2: Latency Metrics
            RowPanel(gridPos=GridPos(h=1, w=24, x=0, y=9),
                     title="Latency Metrics"),

            # Average Latency
            TimeSeries(
                title="Average Prediction Latency",
                dataSource="prometheus",
                targets=[
                    Target(
                        expr='rate(prediction_latency_seconds_sum[1m]) / rate(prediction_latency_seconds_count[1m])',
                        legendFormat='Avg Latency (s)',
                        refId='A',
                    ),
                ],
                gridPos=GridPos(h=8, w=12, x=0, y=10),
                unit="s",
            ),

            # Latency Percentiles
            TimeSeries(
                title="Latency Percentiles",
                dataSource="prometheus",
                targets=[
                    Target(
                        expr='histogram_quantile(0.50, rate(prediction_latency_seconds_bucket[1m]))',
                        legendFormat='p50',
                        refId='A',
                    ),
                    Target(
                        expr='histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[1m]))',
                        legendFormat='p95',
                        refId='B',
                    ),
                    Target(
                        expr='histogram_quantile(0.99, rate(prediction_latency_seconds_bucket[1m]))',
                        legendFormat='p99',
                        refId='C',
                    ),
                ],
                gridPos=GridPos(h=8, w=12, x=12, y=10),
                unit="s",
            ),

            # Row 3: System Metrics
            RowPanel(gridPos=GridPos(h=1, w=24, x=0, y=18),
                     title="System Metrics"),

            # CPU Usage
            TimeSeries(
                title="CPU Usage",
                dataSource="prometheus",
                targets=[
                    Target(
                        expr='system_cpu_usage',
                        legendFormat='CPU %',
                        refId='A',
                    ),
                ],
                gridPos=GridPos(h=8, w=12, x=0, y=19),
                unit="percent",
                valueMin=0,
                valueMax=100,
            ),

            # Memory Usage
            TimeSeries(
                title="Memory Usage",
                dataSource="prometheus",
                targets=[
                    Target(
                        expr='system_memory_usage',
                        legendFormat='Memory %',
                        refId='A',
                    ),
                ],
                gridPos=GridPos(h=8, w=12, x=12, y=19),
                unit="percent",
                valueMin=0,
                valueMax=100,
            ),

            # Row 4: Request Duration Distribution
            RowPanel(gridPos=GridPos(h=1, w=24, x=0, y=27),
                     title="Request Duration Distribution"),

            # Request Duration by Endpoint
            TimeSeries(
                title="Request Duration by Endpoint",
                dataSource="prometheus",
                targets=[
                    Target(
                        expr='rate(starlette_request_duration_seconds_sum[1m]) / rate(starlette_request_duration_seconds_count[1m])',
                        legendFormat='{{method}} {{path_template}}',
                        refId='A',
                    ),
                ],
                gridPos=GridPos(h=8, w=24, x=0, y=28),
                unit="s",
            ),
        ],
    ).auto_panel_ids()

    return dashboard


def save_dashboard(dashboard, output_file="dashboard.json"):
    """Save the dashboard to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(dashboard.to_json_data(), f, indent=2, cls=DashboardEncoder)
    print(f"Dashboard saved to {output_file}")
    print(f"\nTo import into Grafana:")
    print(f"1. Open Grafana UI (http://localhost:3000)")
    print(f"2. Go to Dashboards > Import")
    print(f"3. Upload {output_file}")
    print(f"4. Select your Prometheus data source")


if __name__ == "__main__":
    dashboard = generate_dashboard()
    save_dashboard(dashboard)
