FROM python:3.12-slim
WORKDIR /app
COPY server.py model.pkl prediction.proto requirements.txt /app/
RUN pip install -r requirements.txt
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. prediction.proto
EXPOSE 5000
CMD ["python", "-u", "server.py"]