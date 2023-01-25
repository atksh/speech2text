FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Asia/Tokyo
RUN apt-get update && apt-get install -y software-properties-common tzdata git wget curl build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libgl1-mesa-dev \
    ffmpeg \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/openai/whisper.git 
COPY . .

ENTRYPOINT /bin/bash