import re
import sys
import numpy as np
import os
import datetime

SKIPPED_LINES = 2
TFLOPS_LABEL = "throughput per GPU (TFLOP/s/GPU)"
ELAPSED_TIME_LABEL="elapsed time per iteration (ms)"

ITER_LINE_RE = re.compile(r".*iteration\s+(\d+)\/")
ITER_SPLIT_RE = re.compile(r'\.\.\.+')
ARG_START_RE = re.compile(r'---+ arguments ---+')
ARG_END_RE = re.compile(r'---+ end of arguments ---+')
# reference for parameter parsing
#  > number of parameters on (tensor, pipeline) model parallel rank (6, 0): 8624807936
PARAM_PARSER = re.compile(r'.* parameters on \(tensor, pipeline\) model parallel rank \((\d+), (\d+)\): (\d+)')

def pprint(*args):
    print("  ", *args)

def read_file(file_path):
    with open(file_path) as f:
        return f.readlines()

def parse_args(lines):
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if ARG_START_RE.search(line):
            start_idx = i
        if ARG_END_RE.search(line):
            end_idx = i
            break
    if start_idx is None or end_idx is None:
        return None
    else:
        lines = [line.strip() for line in lines[start_idx+1:end_idx]]
        arguments = {}
        for line in lines:
            if line:
                try:
                    parts = ITER_SPLIT_RE.split(line.split(":")[-1])
                    key = parts[0].strip()
                    value = parts[1].strip()
                    arguments[key] = value
                except:
                    print(f"Failed to grep arguments from line: \"{line}\"")
        return arguments

def parse_iteration(line):
    splits = line.split("|")
    result = {}
    for split in splits[1:-1]:
        key, value = split.split(":")
        key = key.strip()
        value = float(value.strip())
        result[key] = value
    return result

def get_key(key, througput_dict):
    try:
        return [t[key] for t in througput_dict]
    except KeyError:
        return [-1 for t in througput_dict][0]

def extract_params(lines):
    params = [PARAM_PARSER.match(line).groups() for line in lines if PARAM_PARSER.match(line)]
    params = {f"{t}_{p}": int(n) for t, p, n in params}
    params = sum(params.values())
    return params

def extract_values(filepath):
    lines = read_file(filepath)
    args = parse_args(lines)
    # print(args['profile'])
    if args['profile'] != "True":
        throughput = [parse_iteration(line) for line in lines if ITER_LINE_RE.match(line)]
        throughput = throughput[SKIPPED_LINES:]
    else:
        omit_start = int(args['profile_step_start'])
        omit_end = int(args['profile_step_end'])
        throughput = [parse_iteration(line) for line in lines if ITER_LINE_RE.match(line)]
        throughput = throughput[SKIPPED_LINES:omit_start] + throughput[omit_end:]

    num_model_params = extract_params(lines)
    time_created = os.path.getctime(filepath)
    time_created = datetime.datetime.fromtimestamp(time_created).strftime('%Y-%m-%d %H:%M:%S')

    if not args:
        return None
    WORLD_SIZE = int(args["world_size"])
    seq_len = args["seq_length"]
    batch_size = args["global_batch_size"]
    tgs = [int(seq_len)*int(batch_size) / t[ELAPSED_TIME_LABEL]*1000 / WORLD_SIZE  for t in throughput]
    loss = get_key("lm loss", throughput)

    # make loss to be tuple of starting loss and ending loss, including nans
    loss = np.array(loss)
    loss_has_nan = (loss == -1).any()
    if loss_has_nan:
        loss_start = np.nan
        loss_end = np.nan
    else:
        loss_start = loss[0]
        loss_end = loss[-1]
    tflops = get_key(TFLOPS_LABEL, throughput)
    mem_usages = get_key("mem usages", throughput)
    # check if fsdp-key exists
    fsdp_key = 'use_torch_fsdp2'

    if fsdp_key not in args and 'fsdp' not in args:
        args['fsdp'] = False 
    elif fsdp_key in args:
        args['fsdp'] = args[fsdp_key]


    
    result = {
        "tgs": tgs,
        "tflops": tflops,
        "mem_usages": mem_usages,
        "seq_len": seq_len,
        "micro_batch_size": args['micro_batch_size'],
        "batch_size": batch_size,
        "world_size": WORLD_SIZE,
        "data_parallel_size": args['data_parallel_size'],
        "fsdp": args['fsdp'],
        "precision": args['params_dtype'],
        "fp8": args['fp8'],
        "rope_fusion": args['apply_rope_fusion'],
        "loss": (loss_start, loss_end),
        "optimizer": args['optimizer'],
        "embedding_size": args['max_position_embeddings'],
        "transformer_impl": args['transformer_impl'],
        "tensor_model_parallel_size": args['tensor_model_parallel_size'],
        "pipeline_model_parallel_size": args['pipeline_model_parallel_size'],
        "recompute_granularity": args['recompute_granularity'],
        "num_model_params": num_model_params,
        "timestamp": time_created,
        "iters": len(tgs),
        "data_path": args['data_path'],
        "filename": filepath}

    return result, len(tgs)
        
   

def main(argv):
    values, iters = extract_values(argv[1])
     ### print relevant args 
    print()
    print(f"File: {values['filename']}")
    for key, value in values.items():
        if key == 'filename':
            continue
        elif key in ['tgs', 'tflops', 'mem_usages']:
            continue
        elif key == 'num_model_params':
            print(f"{key}: {value/1e9:.2f}B", end=", ")
        else:
            print(f"{key}: {value}", end=", ")

    ### print throughput numbers
    row_format = "{:<10} | {:<10.2f} | {:<10.2f} | {:<10.2f}"
    header_format = "{:<10} | {:<10} | {:<10} | {:<10}"
    header = header_format.format(" ", "TGS", "TFLOPs", "mem usages")
    separator = "-" * len(header)
    if iters > 1:
        print("")
        print(f"Skipped first {SKIPPED_LINES} iterations. Averaging over {iters} iterations")
        print(separator)
        print(header)
        print(separator)
        print(row_format.format("mean", np.mean(values['tgs']), np.mean(values['tflops']), np.mean(values['mem_usages'])))
        print(row_format.format("std", np.std(values['tgs']), np.std(values['tflops']), np.std(values['mem_usages'])))
        print(row_format.format("max", np.max(values['tgs']), np.max(values['tflops']), np.max(values['mem_usages'])))
        print(row_format.format("min", np.min(values['tgs']), np.min(values['tflops']), np.min(values['mem_usages'])))
    else:
        print(f"Found {iters} iters, not enough")
    

if __name__ == "__main__":
    main(sys.argv)