FROM python:3.10 
# currently 3.10. not possible for acoular 22.3 (waiting for new release)

# set the working directory in the container
WORKDIR /data

# copy needed scripts to workingdir
COPY . /acoupipe

# copy the dataset dir containing the main.py file in the workingdir
COPY dataset .  

#https://github.com/numba/numba/issues/4032 -> the numba cache directory
# should be at a writable location when using no 
# root priviliges
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV PIP_ROOT_USER_ACTION=ignore

# install dependencies
RUN /usr/local/bin/python -m pip install --upgrade pip
# echoing pip manager version
RUN bash -c 'echo "$(pip --version)"'
# change directory to acoupipe
RUN cd /acoupipe && pip install ".[full]"
RUN cd /data

# run the main.py script
ENTRYPOINT [ "python", "main.py" ]
