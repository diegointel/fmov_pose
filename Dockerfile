FROM continuumio/miniconda3:latest
RUN apt-get update && apt-get install -y build-essential libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN mkdir fmov

WORKDIR "./fmov"

ADD . .

RUN conda env create -f ./environment.yml