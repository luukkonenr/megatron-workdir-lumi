import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client('s3')

bucket = 'commoncrawl'
prefix = 'contrib/datacomp/DCLM-baseline/global-shard_03_of_10/local-shard_1_of_10/'

response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

for obj in response.get('Contents', []):
    key = obj['Key']
    filename = key.split('/')[-1]
    print(f'Downloading {key} → {filename}')
    s3.download_file(bucket, key, filename)

# from datasets import load_dataset
# import os
# import json

# # Parameters
# dataset_name = "mlfoundations/dclm-baseline-1.0"
# output_dir = "./dclm-baseline-1.0/global-shard_03_of_10"
# output_file = os.path.join(output_dir, "local-shard_1_of_10.jsonl")

# # Create output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# # Stream the dataset and select the 4th global shard (index 3)
# print("Loading and streaming dataset...")
# streamed_dataset = load_dataset(dataset_name, split="train", streaming=True)
# shard = streamed_dataset.shard(num_shards=10, index=3)

# # Save each example to a JSONL file
# print(f"Saving to {output_file} ...")
# with open(output_file, "w", encoding="utf-8") as f:
#     for i, example in enumerate(shard):
#         json.dump(example, f, ensure_ascii=False)
#         f.write("\n")

# print(f"Done! Saved {i + 1} examples to {output_file}")
