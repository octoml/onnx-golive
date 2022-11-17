import argparse
import os

import numpy as np
import torch

from olive.util import benchmark

WARMUP_COUNT = 10
BENCHMARK_COUNT = 100
NLP_INPUT = "Lets benchmark an NLP model in all the frameworks"

def run_torchscript(model_path: str, device: str, warmup_count: int, benchmark_count: int):

    print(f"Device - {device}")
    benchmark_fn = None

    if "distillbert" in model_path.lower():
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        inputs = tokenizer(NLP_INPUT, return_tensors="pt")
        inputs = inputs.to(device)
        # print(inputs)

        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        model = model.to(device)
        model.eval()

        def inference_fn():
            with torch.no_grad():
                model(**inputs)

        benchmark_fn = inference_fn

    elif "bert" in model_path.lower():
        from transformers import BertTokenizer, BertForSequenceClassification

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        inputs = tokenizer(NLP_INPUT, return_tensors="pt")
        inputs = inputs.to(device)
        # print(inputs)

        model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        model = model.to(device)
        model.eval()

        def inference_fn():
            with torch.no_grad():
                model(input_ids=inputs['input_ids'], output_attentions=False, output_hidden_states=False)

        benchmark_fn = inference_fn

    elif "gpt2" in model_path.lower():
        from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        inputs = tokenizer(NLP_INPUT, return_tensors="pt")
        inputs = inputs.to(device)
        # print(inputs)        

        model = GPT2ForSequenceClassification.from_pretrained("gpt2")
        model = model.to(device)
        model.eval()

        def inference_fn():
            with torch.no_grad():
                model(input_ids=inputs['input_ids'], use_cache=False, output_attentions=False, output_hidden_states=False)

        benchmark_fn = inference_fn

    elif os.path.exists(model_path):

        model = torch.jit.load(model_path, map_location=device)
        model.eval()

        input_dtypes = {}
        input_shapes = {}

        if "yolov5" in model_path.lower():
            input_dtypes = {"input": "float32" }
            input_shapes = {"input": [1, 3, 640, 640]}
        elif "resnet" in model_path.lower() or "mobilenet" in model_path.lower():
            input_dtypes = {"input": "float32"}
            input_shapes = {"input": [1, 3, 224, 224]}
        else:
            assert not "Unknown model"

        # create input data
        input_data = {}
        for iname, ishape in input_shapes.items():
            dtype = np.dtype(input_dtypes[iname])
            if np.issubdtype(dtype, np.integer):
                d = np.zeros(ishape, dtype=dtype)
            else:
                d = np.random.uniform(size=ishape).astype(dtype)
            input_data[iname] = d

        inference_data = []
        for key, val in input_data.items():
            input_tensor = torch.from_numpy(val).to(device)
            if "bf16" in model_path.lower():
                input_tensor = input_tensor.bfloat16()
            print(f"Input - Name: {key}, Shape: {input_tensor.shape}, DType: {input_tensor.dtype}")                
            inference_data.append(input_tensor)

        def inference_fn():
            with torch.no_grad():
                model(*inference_data)            

        benchmark_fn = lambda: model(*inference_data)

    return benchmark(benchmark_fn, warmup_count, benchmark_count)


# def export_bf16():
#     import torchvision

#     resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
#     resnet.eval()
#     input_dtypes = {"input": "float32"}
#     input_shapes = {"input": [1, 3, 224, 224]}    

#     input_data = {}
#     for iname, ishape in input_shapes.items():
#         dtype = np.dtype(input_dtypes[iname])
#         if np.issubdtype(dtype, np.integer):
#             d = np.zeros(ishape, dtype=dtype)
#         else:
#             d = np.random.uniform(size=ishape).astype(dtype)
#         input_data[iname] = d    

#     input_tensors = [torch.from_numpy(val) for val in input_data.values()]
#     output = resnet(input_tensors[0])
#     print("Resnet50 FP32:", output.dtype, output.numel())

#     traced_resnet = torch.jit.trace(resnet, input_tensors)
#     print("Resnet50 traced")

#     input_tensors_bf16 = input_tensors[0].bfloat16()
#     resnet_bf16 = resnet.bfloat16()
#     output = resnet_bf16(input_tensors_bf16)
#     print("Resnet50 BF16:", output.dtype, output.numel())    

#     traced_resnet_16 = torch.jit.trace(resnet_bf16, input_tensors_bf16)
#     print("Resnet50 BF16 traced")
#     torch.jit.save(traced_resnet_16, "resnet50.bf16.torchscript")
#     torch.onnx.export(traced_resnet_16, input_tensors_bf16, "resnet.bf16.onnx", verbose=True)    


if __name__ == "__main__":

    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="model file path or (bert or gpt2)")
    parser.add_argument("--device", help=f"device - default {default_device}", default=default_device)
    parser.add_argument("--once", action='store_true')
    args = parser.parse_args()    

    print(f"Torch {torch.__version__}")
    intra_threads, inter_threads = torch.get_num_threads(), torch.get_num_interop_threads(),
    print(f"Torch: intra op threads: {intra_threads}, inter op threads: {inter_threads}")

    run_counts = [0, 1] if args.once else [WARMUP_COUNT, BENCHMARK_COUNT]
    latency, runs = run_torchscript(args.model_path, args.device, *run_counts)
    print(f"Avg latency: {latency:0.2f} ms, {runs} runs")





    
