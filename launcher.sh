export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

python3 -u "$@"