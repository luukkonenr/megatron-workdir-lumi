#!/bin/bash

# MIOPEN needs some initialisation for the cache as the default location
# does not work on LUMI as Lustre does not provide the necessary features.
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export NCCL_NCHANNELS_PER_PEER=32
# Report affinity
#echo "Rank $SLURM_PROCID --> $(taskset -p \$\$)"
# Start conda environment inside the container
$WITH_CONDA

# Set interfaces to be used by RCCL.
# This is needed as otherwise RCCL tries to use a network interface it has
# noa ccess to on LUMI.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
#export NCCL_DMABUF_ENABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1

# The usual PyTorch initialisations (also needed on NVIDIA)
# Note that since we fix the port ID it is not possible to run, e.g., two
# instances via this script using half a node each.
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

python3 -u "$@"