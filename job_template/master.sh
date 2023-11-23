#!/bin/bash

echo "Job running on ..."
hostname

infile=$1
iname=$2
tar xvaf myextrafiles.tar

source environ.sh
echo cat environ.sh
cat environ.sh

echo "##################################"
echo "##################################"
echo "##################################"
echo "Copying sandbox and input file and moving it to input location ..."
#gfal-copy $SANDBOX .
echo xrdcp $IDIR/$infile .
xrdcp $IDIR/$infile .
mkdir -p samples/raw/
mv $infile samples/raw/

mkdir models/
mv FT_best-epoch.pt models/

echo "##################################"
echo "##################################"
echo "##################################"
echo "Whats in the folder and in samples/raw/?"
ls -lhrt
echo " "
ls -lhrt samples/raw/

echo "##################################"
echo "##################################"
echo "##################################"
echo "Untarring deepjet-geometric and torch_cluster"
tar xvaf deepjet-geometric.tgz

#tar xvaf torch_cluster.tgz


ls -lhrt

echo "##################################"
echo "##################################"
echo "##################################"
echo "Sourcing LCG"
source /cvmfs/sft.cern.ch/lcg/views/LCG_102b_cuda/x86_64-centos7-gcc8-opt/setup.sh

#source /cvmfs/sft-nightlies.cern.ch/lcg/views/dev4cuda/latest/x86_64-centos7-gcc8-opt/setup.sh

#export PYTHONPATH=${PWD}:${PYTHONPATH}

cd deepjet-geometric/

export PYTHONPATH=${PWD}:${PYTHONPATH}

cd examples/

echo "##################################"
echo "##################################"
echo "##################################"
echo "Starting training"

#echo "y" | pip3 uninstall torch-cluster



#pip3 install -t tc/ torch_cluster
#cd tc/
#export PYTHONPATH=${PWD}:${PYTHONPATH}
#cd ../

echo "Is torch_cluster there?"
/cvmfs/sft-nightlies.cern.ch/lcg/views/dev4cuda/Wed/x86_64-centos7-gcc8-opt/bin/python3 -c 'import torch_cluster;print(torch_cluster.__file__)'

LAYERS=$(echo $iname | grep -o 'onelayerMLP' &> /dev/null && echo "--one_layer_MLP")

#python3 condor_infer_seedOnly.py --ipath ../../samples/ --opath ../../samples/ --mpath ../../models/
PYTHON_CMD="python3 condor_infer.py --ipath ../../samples/ --opath ../../samples/ --mpath ../../models/ --n_output_nodes 8 ${LAYERS}"
echo "#### Running command"
echo ${PYTHON_CMD}

eval ${PYTHON_CMD}

echo "#### Running finished"
pwd
ls -lhrt 

cd ../../

echo "#### Orig dir"
pwd

ls -lhrt 

ofile=${infile/.root/.h5}

echo "#### Copying out"
echo xrdcp test.h5 root://eosuser.cern.ch/$ODIR/$ofile
xrdcp test.h5 root://eosuser.cern.ch/$ODIR/$ofile

echo "#### Done job"
