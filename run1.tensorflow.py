import sys
import time

import numpy as np
import tensorflow as tf

from olive.util import benchmark

WARMUP_COUNT = 5
BENCHMARK_COUNT = 50


TENSORFLOW_DTYPE_TO_STRING = {
    tf.bfloat16: "bfloat16",
    tf.float16: "float16",
    tf.float32: "float32",
    tf.float64: "float64",
    tf.double: "double",
    tf.int8: "int8",
    tf.int16: "int16",
    tf.int32: "int32",
    tf.int64: "int64",
    tf.uint8: "uint8",
    tf.uint16: "uint16",
    tf.uint32: "uint32",
    tf.uint64: "uint64",
    tf.string: "str",
}

NLP_INPUT = "Hello, my dog is cute"

def run_tf(model_path: str):

    benchmark_fn = None

    if model_path.lower() == "bert":
        from transformers import BertTokenizer, TFBertLMHeadModel

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = TFBertLMHeadModel.from_pretrained("bert-base-uncased")

        inputs = tokenizer(NLP_INPUT, return_tensors="tf")
        benchmark_fn = lambda: model(inputs)

    elif model_path.lower() == "gpt2":
        from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = TFGPT2LMHeadModel.from_pretrained("gpt2")

        inputs = tokenizer(NLP_INPUT, return_tensors="tf")
        benchmark_fn = lambda: model(inputs)

    else:
        tf_model = tf.saved_model.load(model_path)
        inference_fn = tf_model.signatures["serving_default"]

        input_tensors = [
            tensor
            for tensor in inference_fn.inputs
            if tensor.dtype != tf.dtypes.resource
        ]

        input_shapes = {}
        input_dtypes = {}
        for tensor in input_tensors:
            name = tensor.name
            input_shapes.update({name: [dim or -1 for dim in tensor.shape]})
            input_dtypes.update(
                {name: TENSORFLOW_DTYPE_TO_STRING[tensor.dtype]}
            )

        # create input data
        input_data = {}
        for iname, ishape in input_shapes.items():
            shape = list(ishape)
            dtype = np.dtype(input_dtypes[iname])
            if np.issubdtype(dtype, np.integer):
                d = np.zeros(shape, dtype=dtype)
            else:
                d = np.random.uniform(size=shape).astype(dtype)
            input_data[iname] = d

        def sanitize_tensor_name(name: str) -> str:
            colon_index = name.rfind(":")
            if colon_index > 0:
                name = name[:colon_index]
            return name

        inference_data = {}
        for name, value in input_data.items():
            inference_data[sanitize_tensor_name(name)] = tf.constant(value)

        for key, value in inference_data.items():
            print(f"Input - Name: {key}, Shape: {value.shape}, DType: {value.dtype}")

        benchmark_fn = lambda: inference_fn(**inference_data)

    return benchmark(benchmark_fn, WARMUP_COUNT, BENCHMARK_COUNT)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run1_tensorflow.py saved_model_path")
        sys.exit(1)

    intra_threads, inter_threads =  tf.config.threading.get_intra_op_parallelism_threads(), tf.config.threading.get_inter_op_parallelism_threads(),
    print(f"TF: intra op threads: {intra_threads}, inter op threads: {inter_threads}")
    print(f"TF: devices: {tf.config.get_visible_devices()}")

    model_path = sys.argv[1]
    latency, runs = run_tf(model_path)
    print(f"Avg latency: {latency:0.2f} ms, {runs} runs")
