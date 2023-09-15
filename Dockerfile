FROM python:3.8
# Instala Supervisor
RUN apt-get update && apt-get install -y supervisor

# Crea un directorio de trabajo
WORKDIR /app

# Copia los archivos de la aplicación en el contenedor
COPY . /app

# Instala las dependencias de la aplicación
RUN pip install -r requirements.txt

# Configura las variables de entorno para Celery
ENV CELERY_BROKER_URL=amqp://guest:guest@rabbitmq:5672//
ENV CELERY_RESULT_BACKEND=redis://redis:6379/0

# Expon el puerto en el que se ejecuta tu aplicación FastAPI
EXPOSE 8000

# Copia el archivo de configuración de Supervisor al contenedor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Comando para ejecutar Supervisor
CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]


