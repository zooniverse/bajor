FROM python:3.10-slim

ENV LANG=C.UTF-8

WORKDIR /usr/src/bajor

RUN apt-get update && apt-get -y upgrade && \
    apt-get install --no-install-recommends -y \
    build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ARG REVISION=''
ENV REVISION=$REVISION
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/usr/src/bajor

# install dependencies
RUN pip install --upgrade pip
RUN pip install poetry

COPY pyproject.toml ./
COPY poetry.toml ./
COPY poetry.lock ./

# Allow installing dev dependencies to run tests
ARG INSTALL_DEV=false

RUN if [ "${INSTALL_DEV}" = 'true' ]; then poetry install; else poetry install --no-dev; fi

COPY . .

# start the api
CMD ["poetry", "run", "start"]
