#!/bin/bash
#SBATCH --job-name=conv
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --partition=small-g
#SBATCH --time=00:20:00
#SBATCH --gpus-per-node=1
#SBATCH --account=project_462000353
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

### This script launches a Megatron to HuggingFace-conversion for dense llama and mistral models. As Megatron doesn't offer built-in functionality for selecting an intermediate checkpoint,                                         ###
### but selects only the latest based on the value in "checkpoint_dir/latest_checkpointed_iteration.txt", we create a symlinked temporary directory for the selected checkpoint with an appropriate .txt file                        ###
### Container used here has incompatible transformers, so we install it in the user space for this use-case. This installation can be also changed with by setting export PYTHONUSERBASE to some arbitrary path and using that for   ###
### conversions.

## USAGE: sbatch INPUT_DIR OUTPUT_DIR (optional)
## OUTPUT: output_dir/iteration_0000XXXX
## EXAMPLE: sbatch my_run/iteration_000050000 converted_checkpoints

echo "SLURM job started"

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif
export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd

TOKENIZER_DIR=/scratch/project_462000353/models/llama31-8b/  # <----- Change this to your needs

# Check if LOAD_DIR argument is passed
if [ -z "$1" ]; then
  echo "Error: LOAD_DIR is not provided. Exiting."
  exit 1
fi
if [ -z "$2" ]; then
    OUT_DIR="./converted_checkpoints/"
fi

# Input path (update this or pass it as a parameter)
LOAD_DIR=$1
OUT_DIR=$2


# Extract the last segment containing "iter_"
iter_segment=$(echo "$LOAD_DIR" | tr '/' '\n' | grep 'iter_' | tail -n 1)

# Temporary directory setup
uuid=$(uuidgen)
tmp_dir_path=".tmp_checkpoint_$uuid"

# Cleanup function to remove the temp directory if it exists
cleanup() {
  if [ -d "$tmp_dir_path" ]; then
    echo "Cleaning up temporary directory: $tmp_dir_path"
    rm -rf "$tmp_dir_path"
  fi
}

# Ensure cleanup is called on exit, whether successful or due to an error
trap cleanup EXIT

if [[ $iter_segment =~ iter_([0-9]+) ]]; then
    echo "Checkpoint iteration found in load_dir"

    raw_iteration="${BASH_REMATCH[1]}"
    iteration=$((10#$raw_iteration))  # Converts to integer, strips leading zeros

    mkdir -p "$tmp_dir_path" || { echo "Failed to create directory $tmp_dir_path"; exit 1; }

    # Create symlink to the original LOAD_DIR
    ln -s "$(readlink -f "$LOAD_DIR")" "$tmp_dir_path/$iter_segment" || { echo "Failed to create symlink"; exit 1; }

    # Write clean integer to the file
    echo "$iteration" > "$tmp_dir_path/latest_checkpointed_iteration.txt"

    echo "Symlinked $LOAD_DIR to $tmp_dir_path"

    # Update LOAD_DIR variable if needed
    LOAD_DIR="$tmp_dir_path"

    # Set base checkpoint directory and build full path with step
    CHECKPOINT_PATH=$LOAD_DIR

    OUTPUT_ROOT=$OUT_DIR
    OUTPUT_DIR="${OUTPUT_ROOT}/checkpoint_${BASH_REMATCH[1]}"
    

    singularity exec $CONTAINER /bin/bash -c "
      pip install transformers==4.48.2;
      python3 Megatron-LM/tools/checkpoint/convert.py \
        --model-type GPT \
        --loader mcore \
        --saver llama_mistral \
        --load-dir \"$CHECKPOINT_PATH\" \
        --save-dir \"$OUTPUT_DIR\" \
        --tokenizer-dir \"$TOKENIZER_DIR\"
    "
else
    echo "No checkpoint iteration found in load_dir"
fi