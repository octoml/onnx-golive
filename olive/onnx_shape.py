import sys
import onnx

def get_shape(onnx_model: onnx.ModelProto):
    def _get_dim_value(d):
        if (d.HasField("dim_value")):
            return d.dim_value
        elif (d.HasField("dim_param")):
            return d.dim_param
        else:
            return "?"

    inputs = {}
    for input in onnx_model.graph.input:
        tensor_type = input.type.tensor_type
        inputs[input.name] = [_get_dim_value(d) for d in tensor_type.shape.dim]

    outputs = {}
    for output in onnx_model.graph.output:
        tensor_type = input.type.tensor_type
        outputs[output.name] = [_get_dim_value(d) for d in tensor_type.shape.dim]
    return inputs, outputs

if __name__ == '__main__':
    model = onnx.load(sys.argv[1])
    _in, _out = get_shape(model)
    print(_in)
    print(_out)

