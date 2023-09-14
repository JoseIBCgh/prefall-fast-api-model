# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

EXPOSE 8000/tcp

WORKDIR /fastapi-docker

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD ["gunicorn", "-w", "3", "-k", "uvicorn.workers.UvicornWorker", "app:app"]
