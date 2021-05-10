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
COPY README.md .
COPY setup.py .
COPY sampling ./sampling

# copy needed scripts to workingdir
COPY dataset .  

# install dependencies
RUN pip install -r requirements.txt


