FROM nvidia/cuda:12.1.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/src/zoobot

# Install prerequisites and add deadsnakes PPA for Python 3.10
RUN apt-get update && apt-get -y upgrade && \
    apt-get install --no-install-recommends -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python3.10 \
    python3.10-distutils \
    python3.10-dev \
    python3-pip \
    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Link Python 3.10 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# install a newer version of pip
# as we can't use the use the ubuntu package pip version (20.0.2)
# because it doesn't install the gz datasets from github correctly
# use pip (22.1.2) or higher
RUN python -m pip install --upgrade pip
RUN apt-get remove -y python3-pip
RUN ln -s /usr/local/bin/pip3 /usr/bin/pip
RUN ln -s /usr/local/bin/pip3 /usr/bin/pip3


# Install project dependencies (see setup.py)
COPY setup.py .
RUN pip install . --extra-index-url https://download.pytorch.org/whl/cu121