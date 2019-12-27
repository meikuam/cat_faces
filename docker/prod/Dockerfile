FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt install -y --no-install-recommends \
    sudo \
    build-essential \
    pkg-config \
    curl \
    git \
    zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libssl-dev \
    zlib-devel bzip2-devel readline-devel sqlite-devel openssl-devel

ARG UID
ARG USER
ARG PROJECT_NAME

RUN useradd -d /home/$USER -ms /bin/bash  -u $UID -G sudo $USER
RUN echo $USER:$USER | chpasswd
USER $USER

# python
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
ENV PATH /home/$USER/.pyenv/shims:/home/$USER/.pyenv/bin:$PATH
ENV PYENV_SHELL bash

RUN command pyenv rehash 2>/dev/null
RUN git clone https://github.com/momo-lab/pyenv-install-latest.git /home/$USER/.pyenv/plugins/pyenv-install-latest

RUN pyenv install-latest 3.7
RUN pyenv global $(pyenv versions | grep 3.7)

USER root

ENV NOTVISIBLE "in users profile"
EXPOSE 22

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

CMD [ "python", "src/start_bot.py" ]
