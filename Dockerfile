FROM tensorflow/tensorflow:latest-gpu-jupyter AS jupyter

#https://github.com/numba/numba/issues/4032 -> the numba cache directory
# should be at a writable location when using no 
# root priviliges
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV PIP_ROOT_USER_ACTION=ignore

# custom acoular version
RUN pip install acoular

# copy needed scripts to workingdir
COPY . /tmp/acoupipe
RUN cd /tmp/acoupipe

# install
RUN pip install /tmp/acoupipe
#RUN pip install numba tbb
#RUN apt-get update && apt-get install -y --no-install-recommends libtbb-dev

############################################ base builds ###########################################################

FROM python:3.10 AS base
# currently 3.10. not possible for acoular 22.3 (waiting for new release)

#https://github.com/numba/numba/issues/4032 -> the numba cache directory
# should be at a writable location when using no 
# root priviliges
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV PIP_ROOT_USER_ACTION=ignore

# set the working directory in the container
WORKDIR /src
ARG TOKEN

# install dependencies
RUN /usr/local/bin/python -m pip install --upgrade pip
# echoing pip manager version
RUN bash -c 'echo "$(pip --version)"'

# custom acoular version
RUN git clone https://kujawski:${TOKEN}@git.tu-berlin.de/acoular-dev/kujawski/acoular.git /tmp/acoular
RUN pip install /tmp/acoular
RUN rm -r /tmp/acoular

# copy needed scripts to workingdir
COPY . /tmp/acoupipe
RUN cd /tmp/acoupipe

# copy app 
RUN mkdir /app
COPY ./app /app

# install
RUN pip install /tmp/acoupipe
#RUN pip install numba tbb
#RUN apt-get update && apt-get install -y --no-install-recommends libtbb-dev

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
