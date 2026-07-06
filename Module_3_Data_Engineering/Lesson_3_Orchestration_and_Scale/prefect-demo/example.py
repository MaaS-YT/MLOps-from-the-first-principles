from prefect import flow, task
import time

@task(retries=3, retry_delay_seconds=10)
def fetch_data():
    time.sleep(0.2)
    return "data.csv"


@task
def process_data(path: str):
    time.sleep(0.2)
    return "processed.parquet"


@task
def train_model(data: str):
    time.sleep(0.2)
    return "model.pkl"


@task
def deploy_model(model: str):
    time.sleep(0.2)
    print(f"Deployed model: {model}")


@flow(name="30s_model_training")
def training_pipeline():
    raw = fetch_data()
    processed = process_data(raw)
    model = train_model(processed)
    deploy_model(model)

# for local testing
training_pipeline()
