export NUM_EXPERTS=64
export MOE_TOPK=8
export MOE_NUM_SHARED_EXPERTS=0
export MOE_LAYER_FREQ=1
export MICRO_BATCH_SIZE=8

export SAVE_INTERVAL=1000
export EVAL_INTERVAL=1000
export LOG_INTERVAL=1
export SAVE_DIR="checkpoints/MOE_2B_400M/baseline_model"
export LOAD_DIR="${SAVE_DIR}"

export MODEL_NAME="MOE_2B_400M_baseline"
export FILE_DIR="$(dirname "$(realpath "$0")")"

sbatch --nodes 1 --time 0-00:65:00 ${FILE_DIR}/lumi_train.sh