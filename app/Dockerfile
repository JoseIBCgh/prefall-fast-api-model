FROM python:3.8

WORKDIR /app

# Copia los archivos de la aplicación en el contenedor
COPY . /app

# Instala las dependencias de la aplicación
RUN pip install -r requirements.txt

# Configura las variables de entorno para Celery
ENV CELERY_BROKER_URL=amqp://guest:guest@srv.ibc.bio:32837//
ENV CELERY_RESULT_BACKEND=redis://srv.ibc.bio:32828/0

EXPOSE 8000

# Comando para ejecutar la aplicación FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]



