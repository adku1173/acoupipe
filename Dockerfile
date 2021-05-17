# set base image (host OS)
# TODO: Decide between two possibilities: 
#1. include data set files (main.py etc.) into the Dockerimage 
#2: user mounts directory with data set files (main.py etc.) into the Docker Container 

FROM python:3.8
MAINTAINER Adam Kujawski <adam.kujawski@tu-berlin.de>

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

# install dependencies
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# run the main.py script
CMD ["main.py"]
