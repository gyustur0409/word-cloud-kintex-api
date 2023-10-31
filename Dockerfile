FROM openjdk:11

COPY . /app

RUN apt-get update -y
RUN apt-get install -y python3-pip

RUN pip install -r /app/requirements.txt

WORKDIR /app

CMD ["gunicorn","-b",":8080", "app:app"]

EXPOSE 8080