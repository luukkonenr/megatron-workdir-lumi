export NUM_EXPERTS=64
export MOE_TOPK=8
export MOE_NUM_SHARED_EXPERTS=0
export MOE_LAYER_FREQ="[0]*1+[1]*11" # first layer dense, rest moe 
export MICRO_BATCH_SIZE=8

export SAVE_INTERVAL=1000
export EVAL_INTERVAL=1000
export LOG_INTERVAL=1
export SAVE_DIR="checkpoints/MOE_2B_400M/dense_input_layer"
export LOAD_DIR="${SAVE_DIR}"

export MODEL_NAME="MOE_2B_400M_dense_input_layer"
export FILE_DIR="$(dirname "$(realpath "$0")")"

sbatch --nodes 4 ${FILE_DIR}/lumi_train.sh