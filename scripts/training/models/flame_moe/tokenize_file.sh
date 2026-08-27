#!/bin/bash
#SBATCH --job-name=test-tokenize
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --partition=dev-g
#SBATCH --time=02:00:00
#SBATCH --account=project_462000353
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

wd=${PWD}
datadir=${wd}/flame_moe/data
pushd ${wd}/Megatron-LM # temporiraly cd into megatron 
TOKENIZER=EleutherAI/pythia-12b

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif
export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd
export PYTHONUSERBASE=""
echo "Submitting file $1"
for file in $1/*jsonl; do 
    srun --label \
        singularity exec -B $PWD $CONTAINER \
            python3 tools/preprocess_data.py \
            --input $file \
            --output-prefix "${file}_tokenized" \
            --tokenizer-type HuggingFaceTokenizer \
            --tokenizer-model $TOKENIZER \
            --append-eod \
            --workers $SLURM_CPUS_PER_TASK
done
popd
