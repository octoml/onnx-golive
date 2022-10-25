import sys
from typing import Dict

import time
import numpy as np
import onnxruntime as ort
import onnx

def run(
    onnx_model: onnx.ModelProto,
    input_values: Dict[str, np.ndarray],
    ep: str,
    repeats=20,
) -> float:
    session = ort.InferenceSession(onnx_model.SerializeToString(), providers=[ep])

    session.run(None, input_values)

    st = time.time()
    for k in range(repeats):
        session.run(None, input_values)
    return (time.time() - st) / repeats

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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run1.py model.onnx [EP]")
        sys.exit(1)

    onnx_model = onnx.load(sys.argv[1])
    ep = TensorrtExecutionProvider = sys.argv[2] if len(sys.argv) >= 3 else 'TensorrtExecutionProvider'

    input_shapes = get_input_shapes(onnx_model)
    print(f"INPUTS: {input_shapes}")

    input_dict = {shape_name: np.zeros(shape_size, dtype='float32') for (shape_name, shape_size) in input_shapes.items()}
    latency = run(onnx_model, input_dict, ep)
    print(f"Avg latency: {latency * 1000} ms")
