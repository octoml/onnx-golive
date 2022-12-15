
import argparse
import os
import statistics
import subprocess
import time

from typing import Dict
from multiprocessing import Pool

import onnx
import numpy as np
import onnxruntime as ort
from onnxruntime.tools import onnx_model_utils

def flush_cache():
    subprocess.check_call('sync; echo 3 > /proc/sys/vm/drop_caches', shell=True)

def run1(onnx_model_file: str,
         input_values: Dict[str, np.ndarray],
         providers,
         ):
    st = time.time()
    session = ort.InferenceSession(
        onnx_model_file,
        providers=providers
    )

    session.run(None, input_values)
    return time.time() - st

def run_background(
    onnx_model_file: str,
    input_values: Dict[str, np.ndarray],
    providers,
):
    with Pool(processes=1) as pool:
        return pool.apply(run1, args=[onnx_model_file, input_values, providers])

def run_warm(onnx_model_file: str,
             input_values: Dict[str, np.ndarray],
             providers,
             repeats
             ):
    with open(onnx_model_file, 'rb') as fh:
        b = fh.read()
    latencies = [run_background(onnx_model_file, input_values, providers) for x in range(repeats)]
    return statistics.mean(latencies)

def run_cold(onnx_model_file: str,
             input_values: Dict[str, np.ndarray],
             providers,
             repeats
             ):
    latencies = []
    for k in range(repeats):
        flush_cache()
        latencies.append(run_background(onnx_model_file, input_values, providers) for x in range(repeats))
    return statistics.mean(latencies)

def run_hot(
    onnx_model_file: str,
    input_values: Dict[str, np.ndarray],
    providers,
    repeats=20,
):
    print("Loading model data")

    # Load model data into memory
    session = ort.InferenceSession(
        onnx_model_file,
        providers=providers
    )

    session.run(None, input_values)

    print("Begin benchmark")

    st = time.time()
    for k in range(repeats):
        session.run(None, input_values)
    return (time.time() - st) / repeats

RUN_FUNCS = {"hot": run_hot, "cold": run_cold, "warm": run_warm}

def get_input_shapes(onnx_model_file: str):
    onnx_model = onnx.load(onnx_model_file)

    shapes = {}
    for input in onnx_model.graph.input:
        # get type of input tensor
        name = input.name
        tensor_type = input.type.tensor_type

        # check if it has a shape:
        assert tensor_type.HasField("shape")

        shape = []
        for d in tensor_type.shape.dim:
            # the dimension may have a definite (integer) value or a symbolic identifier or neither:
            if (d.HasField("dim_value")):
                shape.append(d.dim_value)
            elif (d.HasField("dim_param")):
                shape.append(d.dim_param)
            else:
                shape.append("?")
        shapes[name] = shape

    return shapes

ONNX_TO_NP_TYPE_MAP = {
    "tensor(bool)": bool,
    "tensor(int)": np.int32,
    'tensor(int32)': np.int32,
    'tensor(int8)': np.int8,
    'tensor(uint8)': np.uint8,
    'tensor(int16)': np.int16,
    'tensor(uint16)': np.uint16,
    'tensor(uint64)': np.uint64,
    "tensor(int64)": np.int64,
    'tensor(float16)': np.float16,
    "tensor(float)": np.float32,
    'tensor(double)': np.float64,
    'tensor(string)': np.string_,
}

def get_input_dict(providers, model, shapes):
    input_dict = {}
    inputs = ort.InferenceSession(model, providers=providers).get_inputs()
    input_types = []

    for i in range(0, len(inputs)):
        if inputs[i].type in ONNX_TO_NP_TYPE_MAP.keys():
            input_types.append(ONNX_TO_NP_TYPE_MAP[inputs[i].type])
        else:
            raise KeyError("failed in mapping operator {} which has type {}".format(
                inputs[i].name, inputs[i].type))

    for i in range(0, len(inputs)):
        shape = [1 if (type(x) is int and x < 0) else x for x in shapes[inputs[i].name]]
        # generate values
        vals = np.random.random_sample(shape).astype(input_types[i])
        input_dict[inputs[i].name] = vals

    return input_dict

PROVIDERS = {
    "cpu": [("CPUExecutionProvider", {})],
    "trt": [('TensorrtExecutionProvider', {"trt_engine_cache_enable": "True", "trt_engine_cache_path": os.getcwd(), "trt_fp16_enable": "True"})],
    "cuda": [("CUDAExecutionProvider", {})],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_repeats", help="Number of repeats", type=int, default=20)
    parser.add_argument("-e", '--ep', help='ONNX execution provider', choices=['cpu', 'trt', 'cuda'], default='cpu')
    parser.add_argument("-c", '--cache', help='cache behavior', choices=['cold', 'warm', 'hot'], default='hot')
    parser.add_argument("model", help="The onnx model to benchmark")
    args = parser.parse_args()
   
    input_shapes = get_input_shapes(args.model)
    print(f"Benchmarking model {args.model} on {args.ep} with inputs: {input_shapes}")

    providers = PROVIDERS[args.ep]

    input_dict = get_input_dict(providers, args.model, input_shapes)
    print({key: val.shape for (key, val) in input_dict.items()})

#    latencies = []
    run_func = RUN_FUNCS[args.cache]
    lat = 1000 * run_func(args.model, input_dict, providers, args.num_repeats)

    print(args.cache, lat)
