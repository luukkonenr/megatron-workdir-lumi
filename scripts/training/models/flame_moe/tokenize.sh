data_root="/pfs/lustrep2/scratch/project_462000353/risto/git_repo_tests/megatron-moe-training/flame_moe/data"

sbatch flame_moe/tokenize_file.sh "${data_root}/shard1"
sbatch flame_moe/tokenize_file.sh "${data_root}/shard2"
# sbatch flame_moe/tokenize_file.sh "${data_root}/shard3"
# sbatch flame_moe/tokenize_file.sh "${data_root}/shard4"
# sbatch flame_moe/tokenize_file.sh "${data_root}/shard5"