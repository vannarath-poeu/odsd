FROM python:3.8

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY . /app

CMD export PYTHONPATH=/app && pytest /app