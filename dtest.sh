HOSTDIR="<path to datasets>"
NTASKS=2

#export PYTHONWARNINGS="ignore::DeprecationWarning"
#export TF_CPP_MIN_LOG_LEVEL=3
#export TF_ENABLE_ONEDNN_OPTS=0
docker run --network=host -it --user "$(id -u)":"$(id -g)" -v $HOSTDIR:/data/datasets 454176e05f56 python main.py --tasks=$NTASKS
