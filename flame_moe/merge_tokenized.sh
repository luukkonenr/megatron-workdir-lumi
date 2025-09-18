wd=${PWD}
datadir=${wd}/flame_moe/data
TOKENIZER=EleutherAI/pythia-12b

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif
export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd
export PYTHONUSERBASE=""
data_dir=$(realpath $1)
echo "merging directory $data_dir"

pushd ${wd}/Megatron-LM 
singularity exec -B $PWD $CONTAINER \
    python3 tools/merge_datasets.py \
    --input $data_dir \
    --output-prefix "$data_dir"_tokenized
popd
