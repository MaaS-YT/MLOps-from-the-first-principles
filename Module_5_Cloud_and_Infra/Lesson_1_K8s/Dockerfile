FROM python:3.12-slim
WORKDIR /app
COPY model.pkl requirements.txt app.py /app/
RUN pip install -r requirements.txt
EXPOSE 80
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]