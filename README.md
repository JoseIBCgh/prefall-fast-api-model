

### Puesta en producción de aplicación prefall en cloud
1 - Webapp
2 - Mysql
3. - Ecosistema: fastapi, rabbitmq, redis, celery.


**Step 1**: Compilar las imágenes previamente y obtendremos Rabbitmq y Redis. También puedes ser tomadas de dockerhub.


**Step 2**: Subida de Rabbitmq


**Step 3**: Subida de Redis.


**Step 4**: Poner el puerto expuesto de Rabbitmq y Redis en la configuración de myapp y celery según los puertos expuestos anteriomente.

https://github.com/IBCBio/prefall-fast-api-model/blob/main/docker-compose.yml

* Cambiar el environment de myapp
```bash
 myapp:
    environment:
      - CELERY_BROKER_URL=amqp://guest:guest@srv.ibc.bio:32837//
      - CELERY_RESULT_BACKEND=redis://srv.ibc.bio:32828/0
```
* Cambiar el environment del celery

```bash
  celery:
    
    environment:
      - CELERY_BROKER_URL=amqp://guest:guest@srv.ibc.bio:32837//
      - CELERY_RESULT_BACKEND=redis://srv.ibc.bio:32828/0
```

**Step 5**: Poner el puerto del Step 4 en el celery:

https://github.com/IBCBio/prefall-fast-api-model/blob/main/celery_worker/celery_worker.py

```
celery.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://srv.ibc.bio:32828/0") 
```

**Step 5**: Ídem que paso 5 pero en la ruta app
https://github.com/IBCBio/prefall-fast-api-model/blob/main/app/celery_worker.py
```
celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "amqp://guest:guest@srv.ibc.bio:32837//")
celery.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://srv.ibc.bio:32828/0") 
```

**Step 5**: Configuración de puertos en el docker file

https://github.com/IBCBio/prefall-fast-api-model/blob/main/app/Dockerfile

```bash
# Configura las variables de entorno para Celery
ENV CELERY_BROKER_URL=amqp://guest:guest@srv.ibc.bio:32837//
ENV CELERY_RESULT_BACKEND=redis://srv.ibc.bio:32828/0
```

**Step 6**: Rebuild de las imágenes y obtendremos la fastapi. Subimos al docker la fastapi y vemos el puerto expuesto.

**Step 7**: Subida de mysql8

Importar prefall.sql en mysql y conocer la cadena de conexión mysql.

**Step 8**: Configuración de la webapp. Cambio de puerto para que utilice el de la fastapi
https://github.com/IBCBio/prefall-webapp-code/blob/main/apps/static/assets/js/predict.js

```bash
...
// Send the AJAX POST request
        $.ajax({
            type: 'POST',
            url: 'http://srv.ibc.bio:32840/predict',
...
function poll(task_id){
    console.log("poll")
    $.getJSON('http://srv.ibc.bio:32840/tasks/<task_id>?task_id=' + task_id, function(data) {
...
```

- Configurar conexión mysql

https://github.com/IBCBio/prefall-webapp-code/blob/main/apps/config.py
```
   SQLALCHEMY_DATABASE_URI = 'mysql://root:root@srv.ibc.bio:32817/prefall'
```
Se regenera la imagen de webapp y se sube




