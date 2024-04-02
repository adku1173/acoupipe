FROM tensorflow/tensorflow:latest-gpu-jupyter AS jupyter-gpu

#https://github.com/numba/numba/issues/4032 -> the numba cache directory
# should be at a writable location when using no 
# root priviliges
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV PIP_ROOT_USER_ACTION=ignore

# custom acoular version
RUN pip install acoular>=24.03

# copy needed scripts to workingdir
COPY . /tmp/acoupipe
RUN cd /tmp/acoupipe

# install
RUN pip install /tmp/acoupipe

############################################ base builds ###########################################################

FROM python:3.11 AS base

#https://github.com/numba/numba/issues/4032 -> the numba cache directory
# should be at a writable location when using no 
# root priviliges
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV PIP_ROOT_USER_ACTION=ignore

# set the working directory in the container
WORKDIR /src

# install dependencies
RUN /usr/local/bin/python -m pip install --upgrade pip
# echoing pip manager version
RUN bash -c 'echo "$(pip --version)"'

# custom acoular version
#ARG TOKEN
#RUN git clone https://kujawski:${TOKEN}@git.tu-berlin.de/acoular-dev/kujawski/acoular.git /tmp/acoular
#RUN pip install /tmp/acoular
#RUN rm -r /tmp/acoular

RUN pip install acoular>=24.03

# copy needed scripts to workingdir
COPY . /tmp/acoupipe
RUN cd /tmp/acoupipe

# copy app 
RUN mkdir /app
COPY ./app /app

# install
RUN pip install /tmp/acoupipe

# run the main.py script to save data to file
CMD [ "python", "/app/main.py" ]

# second stage
FROM base AS full
RUN cd /tmp/acoupipe
RUN pip install "/tmp/acoupipe[full]"

# third stage
FROM full AS dev
RUN cd /tmp/acoupipe
RUN pip install "/tmp/acoupipe[dev]"
