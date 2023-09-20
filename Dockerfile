FROM python:3.8
LABEL org.opencontainers.image.authors="adam.kujawski@tu-berlin.de"

# set the working directory in the container
WORKDIR /data

# copy the dependencies file to the working directory
COPY requirements.txt .
COPY README.rst .
##COPY setup.py .

# copy needed scripts to workingdir
COPY acoupipe ./acoupipe
COPY dataset .  

#https://github.com/numba/numba/issues/4032 -> the numba cache directory
# should be at a writable location when using no 
# root priviliges
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
#ENV PYTHONWARNINGS="ignore::DeprecationWarning"
ENV PYTHONWARNINGS="ignore,default:::traits"
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0

# install dependencies
RUN apt-get update
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install numba tbb
RUN apt-get install -y --no-install-recommends libtbb-dev

# run the main.py script
CMD ["main.py"]
