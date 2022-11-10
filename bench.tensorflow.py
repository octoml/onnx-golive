import argparse
import sys

import numpy as np
import tensorflow as tf

from olive.util import benchmark

WARMUP_COUNT = 10
BENCHMARK_COUNT = 100

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

NLP_INPUT = "Lets benchmark an NLP model in all the frameworks"

def run_tf(model_path: str, warmup_count: int, benchmark_count: int):

    benchmark_fn = None

    if "distillbert" in model_path.lower():
        from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        inputs = tokenizer(NLP_INPUT, return_tensors="tf")
        # print(inputs)

        model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        benchmark_fn = lambda: model(**inputs)    

    elif "bert" in model_path.lower():
        from transformers import BertTokenizer, TFBertForSequenceClassification

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        inputs = tokenizer(NLP_INPUT, return_tensors="tf")
        # print(inputs)

        model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
        benchmark_fn = lambda: model(input_ids=inputs['input_ids'], output_attentions=False, output_hidden_states=False)

    elif "gpt2" in model_path.lower():
        from transformers import GPT2Tokenizer, TFGPT2ForSequenceClassification

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        inputs = tokenizer(NLP_INPUT, return_tensors="tf")
        # print(inputs)

        model = TFGPT2ForSequenceClassification.from_pretrained("gpt2")
        benchmark_fn = lambda: model(input_ids=inputs['input_ids'], use_cache=False, output_attentions=False, output_hidden_states=False)

    else:
        tf_model = tf.saved_model.load(model_path)
        inference_fn = tf_model.signatures["serving_default"]

        input_shapes = {}
        input_dtypes = {}        

        input_tensors = [
            tensor
            for tensor in inference_fn.inputs
            if tensor.dtype != tf.dtypes.resource
        ]        
        for tensor in input_tensors:
            name = tensor.name
            print(f"Serving Input Def: {name}, {tensor.shape}, {tensor.dtype}")
            if "resnet" in model_path.lower() or "mobilenet" in model_path.lower():
                input_shapes.update({name: [1, 224, 224, 3]})
            else:
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

    return benchmark(benchmark_fn, warmup_count, benchmark_count)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="model file path or (bert or gpt2)")
    parser.add_argument("--once", action='store_true')
    args = parser.parse_args()

    intra_threads, inter_threads =  tf.config.threading.get_intra_op_parallelism_threads(), tf.config.threading.get_inter_op_parallelism_threads(),
    print(f"TF: intra op threads: {intra_threads}, inter op threads: {inter_threads}")
    print(f"TF: devices: {tf.config.get_visible_devices()}")

    run_counts = [0, 1] if args.once else [WARMUP_COUNT, BENCHMARK_COUNT]
    latency, runs = run_tf(args.model_path, *run_counts)
    print(f"Avg latency: {latency:0.2f} ms, {runs} runs")
