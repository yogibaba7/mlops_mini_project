# base model
FROM python:3.11-slim

# workdir
WORKDIR /app

# COPY
COPY API/ ./API/

# run
RUN pip install -r API/requirements.txt

# expose port
EXPOSE 8000

# run FastAPI
CMD ["uvicorn", "API.app:app", "--host", "0.0.0.0", "--port", "8000"]