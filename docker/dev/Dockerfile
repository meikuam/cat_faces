FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive
RUN /bin/sed -i 's/archive.ubuntu.com/mirror.yandex.ru/g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common

RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update \
    && apt install -y --no-install-recommends \
    libsm6 libxext6 libxrender-dev \
    sudo \
    openssh-server \
    build-essential \
    pkg-config \
    vim \
    git \
    curl \
    libgl1-mesa-glx \
    libsqlite3-dev libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev  llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev

ARG UID
ARG USER
ARG PROJECT_NAME
ARG JUPUTER_PORT

RUN useradd -d /home/$USER -ms /bin/bash  -u $UID -G sudo $USER
RUN echo $USER:$USER | chpasswd
USER $USER

# python
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
ENV PATH /home/$USER/.pyenv/shims:/home/$USER/.pyenv/bin:$PATH
ENV PYENV_SHELL bash

RUN command pyenv rehash 2>/dev/null
RUN git clone https://github.com/momo-lab/pyenv-install-latest.git /home/$USER/.pyenv/plugins/pyenv-install-latest

RUN pyenv install-latest 3.6
RUN pyenv global $(pyenv versions | grep 3.6)
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

USER root

ENV NOTVISIBLE "in users profile"
EXPOSE 22
EXPOSE 5000
EXPOSE 6006
EXPOSE $JUPUTER_PORT


RUN mkdir /var/run/sshd

RUN sed -i 's/#PermitRootLogin /PermitRootLogin yes #/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication/PasswordAuthentication yes#/' /etc/ssh/sshd_config
#SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN echo "export PATH=$PATH" >> /etc/profile
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> /etc/profile
RUN echo "export VISIBLE=now" >> /etc/profile
RUN echo "export NOTVISIBLE=$NOTVISIBLE" >> /etc/profile
RUN echo "export LC_ALL=C.UTF-8" >> /etc/profile
RUN echo "export LANG=C.UTF-8" >> /etc/profile

USER $USER
WORKDIR /tmp/
ADD requirements.txt /tmp/requirements.txt

RUN pip --no-cache-dir install  Cython \
    && pip --no-cache-dir install --upgrade pip \
    && pip --no-cache-dir install -r requirements.txt


RUN mkdir /home/$USER/$PROJECT_NAME
WORKDIR /home/$USER/$PROJECT_NAME
