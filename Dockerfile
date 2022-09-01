FROM python:3.9.9

WORKDIR /opt/server

RUN pip install pipenv

COPY Pipfile Pipfile.lock /opt/server/

RUN pipenv install --system --deploy
