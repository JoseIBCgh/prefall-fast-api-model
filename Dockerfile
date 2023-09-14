# syntax=docker/dockerfile:1

 

FROM python:3.8-slim-buster

EXPOSE 8000

WORKDIR /fastapi-docker

 

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

 

COPY . .

 

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
