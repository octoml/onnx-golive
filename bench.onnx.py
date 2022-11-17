import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort

from olive.constants import ONNX_TO_NP_TYPE_MAP
from olive.util import benchmark

WARMUP_COUNT = 10
BENCHMARK_COUNT = 100
NLP_INPUT = "Hello, my dog is cute"

def run(
    model_path: str,
    ep: str,
    warmup_count: int,
    benchmark_count: int,
) -> float:

    onnx_model = onnx.load(model_path)

    session_options = ort.SessionOptions()
    # session_options.enable_profiling = False
    # session_options.log_severity_level = 2
    # session_options.log_verbosity_level = 0
    # session_options.intra_op_num_threads = 8
    session_inputs = {}
    session_outputs = None
    session = ort.InferenceSession(onnx_model.SerializeToString(), session_options, providers=[ep])

    input_defs = session.get_inputs()
    input_names, input_types, input_dims = zip(*[(i.name, ONNX_TO_NP_TYPE_MAP[i.type], i.shape) for i in input_defs])
    print(f"INPUT NAMES: {input_names}, TYPES: {input_types}, SHAPES: {input_dims}")

    output_defs = session.get_outputs()
    output_names, output_types, output_dims = zip(*[(o.name, ONNX_TO_NP_TYPE_MAP[o.type], o.shape) for o in output_defs])
    print(f"OUTPUT NAMES: {output_names}, TYPES: {output_types}, SHAPES: {output_dims}")

    model_filename = os.path.basename(model_path).lower()
    if "bert" in model_filename:
        from transformers import BertTokenizer

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        inputs = tokenizer(NLP_INPUT, return_tensors="np")
        session_inputs = dict(inputs)
    elif "gpt2" in model_filename:
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        inputs = tokenizer(NLP_INPUT, return_tensors="np")
        session_inputs = dict(inputs)
    else:
        # Prepare inputs
        for i in range(0, len(input_defs)):
            # regard unk__32 and None as 1
            shape = [1 if (x is None or (type(x) is str)) else x for x in input_dims[i]]
            vals = np.random.random_sample(shape).astype(input_types[i])
            session_inputs[input_names[i]] = vals

    benchmark_fn = lambda: session.run(session_outputs, session_inputs)
    return benchmark(benchmark_fn, warmup_count, benchmark_count)


if __name__ == "__main__":

    available_eps = set(ort.get_available_providers())
    prioritized_eps = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'DnnlExecutionProvider', 'CPUExecutionProvider']
    default_ep = None
    for e in prioritized_eps:
        if e in available_eps:
            default_ep = e
            break

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="model file path or (bert or gpt2)")
    parser.add_argument("--ep", "--execution_provider", help=f"execution providers {available_eps}", default=default_ep)
    parser.add_argument("--once", action='store_true')
    args = parser.parse_args()

    print(f"Available EPs: {available_eps}")
    print(f"Selected EP: {args.ep}")

    run_counts = [0, 1] if args.once else [WARMUP_COUNT, BENCHMARK_COUNT]
    latency, runs = run(args.model_path, args.ep, *run_counts)
    print(f"Avg latency: {latency:0.2f} ms, {runs} runs")
