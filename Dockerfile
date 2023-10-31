FROM openjdk:8

COPY . /app

RUN apt-get update -y
RUN apt-get install -y python3-pip

RUN pip install -r /app/requirements.txt

RUN apt-get install -y curl
RUN cd /tmp && \
    curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh | bash -s

WORKDIR /app

CMD ["gunicorn","-b",":8080", "app:app"]

EXPOSE 8080