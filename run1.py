import sys
import time

import numpy as np
import onnx
import onnxruntime as ort

from olive.constants import ONNX_TO_NP_TYPE_MAP

WARMUP_COUNT = 1
BENCHMARK_COUNT = 10

def run(
    onnx_model: onnx.ModelProto,
    ep: str,
    warmup_count=WARMUP_COUNT,
    benchmark_count=BENCHMARK_COUNT,
) -> float:

    session_options = ort.SessionOptions()
    # session_options.log_severity_level = 2
    # session_options.log_verbosity_level = 0
    # session_options.intra_op_num_threads = 8
    session = ort.InferenceSession(onnx_model.SerializeToString(), session_options, providers=[ep])
    print(f"SELECTED EPs: {session.get_providers()}")

    # Prepare inputs
    inputs = session.get_inputs()
    input_names, input_types, input_dims = zip(*[(i.name, ONNX_TO_NP_TYPE_MAP[i.type], i.shape) for i in inputs])
    print(f"INPUT NAMES: {input_names}, TYPES: {input_types}, SHAPES: {input_dims}")

    input_values = {}
    for i in range(0, len(inputs)):
        # regard unk__32 and None as 1
        shape = [1 if (x is None or (type(x) is str)) else x for x in input_dims[i]]
        vals = np.random.random_sample(shape).astype(input_types[i])
        input_values[input_names[i]] = vals

    for _ in range(warmup_count):
        session.run(None, input_values)

    st = time.time()
    for _ in range(benchmark_count):
        session.run(None, input_values)
    return ((time.time() - st) / benchmark_count, benchmark_count)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run1.py model.onnx [EP]")
        sys.exit(1)

    onnx_model = onnx.load(sys.argv[1])
    available_eps = ort.get_available_providers()
    print(f"Available EPs: {available_eps}")
    ep = sys.argv[2] if len(sys.argv) >= 3 else 'TensorrtExecutionProvider'

    latency, runs = run(onnx_model, ep)
    print(f"Avg latency: {latency * 1000:0.2f} ms, {runs} runs")
