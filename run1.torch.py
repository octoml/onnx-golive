import os
import sys
import time

import numpy as np
import torch

from olive.util import benchmark

WARMUP_COUNT = 2
BENCHMARK_COUNT = 10

NLP_INPUT = "Hello, my dog is cute"

def run_torchscript(model_path: str, device):

    print(f"Device - {device}")
    benchmark_fn = None

    if "bert" in model_path:
        from transformers import BertTokenizer, BertLMHeadModel

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        inputs = tokenizer(NLP_INPUT, return_tensors="pt")

        model = BertLMHeadModel.from_pretrained("bert-base-uncased")
        model = model.to(device)
        benchmark_fn = lambda: model(**inputs, labels=inputs["input_ids"])

    elif "gpt2" in model_path:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        inputs = tokenizer(NLP_INPUT, return_tensors="pt")

        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model = model.to(device)
        benchmark_fn = lambda: model(**inputs, labels=inputs["input_ids"])

    elif os.path.exists(model_path):

        model = torch.jit.load(model_path, map_location=device)
        model.eval()

        input_dtypes = {}
        input_shapes = {}

        if "yolov5" in model_path.lower():
            input_dtypes = {"input": "float32" }
            input_shapes = {"input": [1, 3, 640, 640]}
        else:
            assert not "Unknown model"

        # create input data
        input_data = {}
        for iname, ishape in input_shapes.items():
            dtype = np.dtype(input_dtypes[iname])
            if np.issubdtype(dtype, np.integer):
                d = np.zeros(shape, dtype=dtype)
            else:
                d = np.random.uniform(size=ishape).astype(dtype)
            input_data[iname] = d

        for key, value in input_data.items():
            print(f"Input - Name: {key}, Shape: {value.shape}, DType: {value.dtype}")

        inference_data = []
        for val in input_data.values():
            input_tensor = torch.from_numpy(val).to(device)
            inference_data.append(input_tensor)

        benchmark_fn = lambda: model(*inference_data)

    return benchmark(benchmark_fn, WARMUP_COUNT, BENCHMARK_COUNT)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run1_tensorflow.py saved_model_path")
        sys.exit(1)

    print(f"Torch {torch.__version__}")
    intra_threads, inter_threads = torch.get_num_threads(), torch.get_num_interop_threads(),
    print(f"Torch: intra op threads: {intra_threads}, inter op threads: {inter_threads}")

    model_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 3 else "cuda" if torch.cuda.is_available() else "cpu"

    latency, runs = run_torchscript(model_path, device)
    print(f"Avg latency: {latency:0.2f} ms, {runs} runs")
