
import argparse
import os
import statistics
import sys
from typing import Dict
import time

import numpy as np
import onnxruntime as ort
import onnx
import json
from onnxruntime.tools import onnx_model_utils
import pandas as pd

def run(
    onnx_model: onnx.ModelProto,
    input_values: Dict[str, np.ndarray],
    ep: str,
    num_threads: int,
    iterations=250,
) -> float:
    options = ort.SessionOptions()

    if num_threads > 0:
        options.intra_op_num_threads=num_threads
    
    session = ort.InferenceSession(
        onnx_model.SerializeToString(),
        options,
        providers=[ep],
    )

    for k in range(5):
        session.run(None, input_values)

    st = time.time()
    for k in range(iterations):
        session.run(None, input_values)
    return (time.time() - st) / iterations

def get_input_shapes(onnx_model: onnx.ModelProto):
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

def get_input_dict(ep, model, shapes):
    input_dict = {}
    inputs = ort.InferenceSession(model.SerializeToString(), providers=[ep]).get_inputs()
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--ep", help="Execution Provider", default='CPUExecutionProvider')
    parser.add_argument("-n", "--num_repeats", help="Number of repeats", type=int, default=10)
    parser.add_argument("-i", "--num_iterations", help="Number of iterations per repeat", type=int, default=200)
    parser.add_argument("-t", "--num_threads", help="Number of intra_op threads", type=int, nargs="*", default=[1])
    parser.add_argument("-s", "--shape", help="Input shape")
    parser.add_argument("model", help="The onnx model to benchmark")
    args = parser.parse_args()

    onnx_model = onnx.load(args.model)

    if args.shape:
        shape_dict = json.loads(args.shape)
        for input_name, input_shape in shape_dict.items():
            print(f"Mapping input shape {input_name} to {input_shape}")
            onnx_model_utils.make_input_shape_fixed(onnx_model.graph,
                                                    input_name, input_shape)
    
    input_shapes = get_input_shapes(onnx_model)
    print(f"Benchmarking model {args.model} with inputs: {input_shapes}")

    input_dict = get_input_dict(args.ep, onnx_model, input_shapes)
    print({key: val.shape for (key, val) in input_dict.items()})
    results = []

    default_affinity = os.sched_getaffinity(os.getpid())

    for t in args.num_threads:
        print("threads=%d" % t)
        if t == 0:
            os.sched_setaffinity(os.getpid(), {0})

        latencies = []
        for k in range(args.num_repeats):
            lat = 1000 * run(onnx_model, input_dict, args.ep, t, args.num_iterations)
            latencies.append(lat)

        if t == 0:
            os.sched_setaffinity(os.getpid(), default_affinity)

        mean = statistics.mean(latencies)
        stdev = statistics.stdev(latencies)
        cov = stdev / mean
        results.append((t, mean, stdev, cov, min(latencies), max(latencies)))

    df = pd.DataFrame(results, columns=['num_threads', 'mean', 'stdev', 'cov', 'min', 'max'])
    print(df.to_csv(path_or_buf=None, header=True, index=False, encoding='utf-8'))
