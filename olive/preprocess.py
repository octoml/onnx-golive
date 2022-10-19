import itertools
import os
import pathlib
import sys
import tempfile

import onnx
from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnxmltools.utils import load_model, save_model

from onnxruntime.tools import onnx_model_utils

import torch
import pandas as pd

from olive.optimization_config import OptimizationConfig
from olive.optimize import optimize

def get_input_shape(model, input_name: str):
    # iterate through inputs of the graph
    for _input in model.graph.input:
        if _input.name != input_name:
            continue

        tensor_type = _input.type.tensor_type
        # check if it has a shape:
        if (tensor_type.HasField("shape")):
            # iterate through dimensions of the shape:
            shape = []
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if (d.HasField("dim_value")):
                    shape.append(d.dim_value)
                elif (d.HasField("dim_param")):
                    shape.append(d.dim_param)
                else:
                    shape.apend("?")
            return shape
        else:
            raise ValueError("Input has no shape")
    else:
        raise ValueError("No such input: " + input_name)

def noop_transform(model):
    return model

def fp16_transform(model):
     convert_float_to_float16(model)
     return model

def batch_size_transform(input_name: str, input_index: int, batch_size: int):
    def _transform(model):
        shape = get_input_shape(model, input_name)
        shape[input_index] = batch_size
        onnx_model_utils.make_input_shape_fixed(model.graph, input_name, shape)
        return model

    return _transform

def get_configurations(input_name, input_index, batch_sizes):
    dims = [
        [('FP16=DISABLED', noop_transform), ('FP16=ENABLED', fp16_transform)],
        [(f"BATCH_SIZE={x}", batch_size_transform(input_name, input_index, x)) for x in  batch_sizes],
    ]

    names = [[x[0] for x in dim] for dim in dims]
    funcs = [[x[1] for x in dim] for dim in dims]
    
    all_funcs = itertools.product(*funcs)
    all_names = itertools.product(*names)

    for func_list, name_list in zip(all_funcs, all_names):
        yield (func_list, name_list)

def write_csv_summary(result_dir):
    all_dfs = []

    for sub_dir in os.listdir(result_dir):
        result_path = os.path.join(result_dir, sub_dir, 'olive_result.csv')
        if not os.path.exists(result_path):
            print("Skipping non-existing result from: " + result_path)
            continue

        print("Reading results from: " + result_path)
        with open(result_path) as fh:
            df = pd.read_csv(fh)
            num_columns = len(df.columns)
            for idx, dimension in enumerate(sub_dir.split('|')):
                print("  Adding column " + dimension)
                
                dim_name, dim_value = dimension.split('=')
                df.insert(num_columns + idx, dim_name, dim_value)
                all_dfs.append(df)

    combined_csv = pd.concat(all_dfs)
    out_file = os.path.join(result_dir, 'merged.csv')

    print("Writing results to: " + out_file)
    combined_csv.to_csv(os.path.join(out_file, index=True, encoding='utf-8'))

def main(model_path, result_dir, input_name, batch_index, batch_sizes):
    providers_list = ['cpu','cuda','tensorrt'] if torch.cuda.is_available() else ['cpu', 'openvino']

    for func_list, name_list in get_configurations(input_name, batch_index, batch_sizes):
        config_name = '|'.join(name_list)
        print("Running configuration: " + config_name)

        out_dir = os.path.join(result_dir, config_name)        
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        
        onnx_model = load_model(model_path)
        
        for name, func in zip(name_list, func_list):
            print('  Applying function ' + name)
            onnx_model = func(onnx_model)

        converted_model_path = os.path.join(out_dir, 'converted_model.onnx')
        save_model(onnx_model, converted_model_path)

        opt_config = OptimizationConfig(
            model_path=converted_model_path,
            providers_list=providers_list,
            run_all=True,
            result_path=out_dir,
        )
        optimize(opt_config)

    write_csv_summary(result_dir)

if __name__ == '__main__':
    model_path = sys.argv[1]
    out_dir = model_path + ".OUT"
    main(model_path, out_dir, "input", 0, [1, 4])

