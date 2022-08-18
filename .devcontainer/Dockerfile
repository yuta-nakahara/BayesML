FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y build-essential checkinstall libssl-dev libffi-dev libbz2-dev libdb-dev libnss3-dev \
    libreadline-dev libgdbm-dev liblzma-dev libncursesw5-dev libsqlite3-dev zlib1g-dev uuid-dev tk-dev
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN apt-get install -y tzdata
RUN apt-get install -y git
RUN apt-get install -y nano
RUN apt-get install -y curl
RUN apt-get install -y wget
RUN apt-get install -y tree
RUN apt-get install -y tmux 
RUN apt-get install -y htop 
RUN apt-get install -y bmon
RUN apt-get install -y iotop
RUN apt-get install -y ca-certificates
RUN apt-get install -y tar
RUN apt-get install -y bzip2
RUN apt-get install -y xz-utils
RUN apt-get install -y unzip
RUN apt-get install -y make
RUN apt-get install -y cpp gcc g++ cmake
RUN apt-get install -y python3-pip python3-dev python3-venv

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install jedi flake8 autopep8
RUN pip3 install jupyter

ARG USER
RUN useradd -m -s /bin/bash ${USER}
USER ${USER}

WORKDIR /home/${USER}

RUN mkdir BayesML
COPY BayesML/ BayesML/
RUN cd BayesML && pip3 install -e .