
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

SHELL [ "/bin/bash", "--login", "-c" ]f

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub

RUN apt-get update

RUN apt-get install --no-install-recommends -y python3-pip python3.8-dev && \
    apt-get install -y vim && \
    apt-get install -y git && \
    apt-get install -y wget


COPY environment.yml /tmp/

ENV MINICONDA_VERSION latest
ENV CONDA_DIR /home/janischl/zampal/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile


RUN conda init bash

ENV PROJECT_DIR /workspace/

WORKDIR $PROJECT_DIR

ENV ENV_PREFIX /workspace/env  
RUN conda env create --prefix $ENV_PREFIX --file /tmp/environment.yml --force && \ 
    conda activate /workspace/env  && \
    conda install mkl

RUN echo "conda activate /workspace/env" >> ~/.bashrc
# COPY ./backend/set_path.py  $PROJECT_DIR
# COPY ./backend/  $PROJECT_DIR 
# COPY ./.git /workspace/.git

# RUN python set_path.py
ENV FLASK_ENV=development
ENV DEBUG=true
COPY gunicorn_config.py /$PROJECT_DIR/gunicorn_config.py 
COPY requirements.txt /$PROJECT_DIR/

RUN apt-get -o Dpkg::Options::='--force-confmiss' install --reinstall -y netbase
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install build-essential
ENV CUDA_HOME=/usr/local/cuda


RUN pip install -r requirements.txt 

RUN conda activate $ENV_PREFIX && \ 
    pip install Werkzeug==0.16.0 && \
    conda env list &&\
    conda install gunicorn && \
    pip install eventlet==0.30.2
    
RUN conda activate $ENV_PREFIX && \ 
    pip install natsort && \ 
    pip install markupsafe==2.0.1 && \
    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch  && \ 
    conda install faiss-gpu cudatoolkit=10.2 -c pytorch --yes && \
    pip install  tensorboard torch-summary torch_optimizer scikit-learn matplotlib seaborn requests ipdb flake8 pyyaml natsort imutils ninja cffi pycocotools 

RUN pip install -U numpy
RUN pip install -r requirements.txt 
RUN pip install -U numpy

EXPOSE 8886

CMD tail -f /dev/null

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

#Run this in the attached shell
# CMD CUDA_VISIBLE_DEVICES=0 gunicorn -c webserver/gunicorn_config.py webserver:app --no-sendfile
