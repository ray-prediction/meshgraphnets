FROM nvidia/cuda:12.8.0-devel-ubuntu20.04
WORKDIR /test
EXPOSE 5000 5001

# RUN apt update

# Install Python 3.6
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y python3.6 python3.6-venv python3.6-dev

# venv
RUN python3.6 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip

COPY meshgraphnets/requirements.txt .
RUN pip install -r requirements.txt
COPY meshgraphnets /test/


# FROM nvidia/cuda:12.8.0-devel-ubuntu20.04
# WORKDIR /test

# RUN apt update && apt upgrade -y
# RUN apt install python3 python3-pip -y
# CMD ["python3", "--version"]

# Install Python 3.6
# RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common
# RUN add-apt-repository ppa:deadsnakes/ppa -y
# RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y python3.6 python3.6-venv python3.6-dev
# RUN curl -sS https://bootstrap.pypa.io/pip/3.6/get-pip.py | python3.6
# # venv
# RUN python3.6 -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# COPY meshgraphnets/requirements.txt .
# RUN pip3 install -r requirements.txt -test
# COPY meshgraphnets /test/
