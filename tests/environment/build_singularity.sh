#!/bin/sh	

#sudo singularity build acoupipe_test.sif singularity.def
singularity build acoupipe.sif docker://adku1173/acoupipe:dev-full
