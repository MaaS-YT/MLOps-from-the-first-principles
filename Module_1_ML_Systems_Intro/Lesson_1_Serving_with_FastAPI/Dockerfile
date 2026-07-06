FROM python:3.11-slim@sha256:f6a8f1e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0
WORKDIR /app 
COPY app.py iris_model.pkl requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt 
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]