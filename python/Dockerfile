FROM python:3.8.6-slim
# https://hub.docker.com/r/nvidia/cuda/
# FROM nvidia/cuda:11.1.1-devel
# FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
# FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
# FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel


## The MAINTAINER instruction sets the author field of the generated images.
LABEL maintainer="wenh06@gmail.com"

## DO NOT EDIT the 3 lines.
RUN mkdir /dgne
COPY ./ /dgne
WORKDIR /dgne

# submodule
# RUN apt-get update && \
#     apt-get upgrade -y && \
#     apt-get install -y git
# RUN git submodule update --init --remote --recursive --merge --progress
# RUN git submodule update --remote --recursive --merge --progress

## Install your dependencies here using apt install, etc.
# RUN apt update && apt upgrade -y && apt clean
# RUN apt install -y python3.8 python3.8-dev python3.8-distutils python3-pip

# latest version of biosppy uses opencv
# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt update
RUN apt install ffmpeg libsm6 libxext6 vim  -y

# RUN apt update && apt install -y --no-install-recommends \
#         build-essential \
#         curl \
#         software-properties-common \
#         unzip

# RUN apt install python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip
# RUN pip install --upgrade pip

# http://mirrors.aliyun.com/pypi/simple/
# http://pypi.douban.com/simple/
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip install torch
## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt


RUN python python/docker_test.py
