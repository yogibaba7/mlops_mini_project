# base model
FROM python:3.11-slim

# workdir
WORKDIR /app

# copy
COPY API/ ./API/

# run
RUN pip install -r API/requirements.txt

# expose
EXPOSE 8000

# command
CMD ["uvicorn", "API.app:app", "--host", "0.0.0.0", "--port", "8000"]