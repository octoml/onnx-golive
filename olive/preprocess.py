import argparse
import datetime
import itertools
import json
import logging
import os
import pathlib
import sys
import tempfile

import smart_open

import onnx
from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnxmltools.utils import load_model, save_model

import onnxruntime as ort
from onnxruntime.tools import onnx_model_utils

from olive.optimization_config import OptimizationConfig
from olive.optimize import optimize
from olive.merge_outputs import write_csv_summary
from olive.instance_type import get_instance_type

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def get_configurations(rewrite_config):
    # A list of "dimensions", each containing a list of tuples of the form:
    # ("feature_name=feature_value", Callable)
    dims = []
    if 'batch_input_name' in rewrite_config:
        batch_input_name = rewrite_config['batch_input_name']
        batch_input_index = rewrite_config['batch_input_index']
        batch_sizes = rewrite_config['batch_sizes']

        logger.info(f"Testing {batch_input_name}:{batch_input_index} over batch sizes: {batch_sizes}")
        dims.append(
            [(f"BATCH_SIZE={x}",
              batch_size_transform(batch_input_name, batch_input_index, x)) for x in  batch_sizes])
    else:
        dims.append([("BATCH_SIZE=1", noop_transform)])

    if 'test_fp16' in rewrite_config:
        logger.info("Enabling fp16 testing")
        dims.append(
            [('FP16=DISABLED', noop_transform), ('FP16=ENABLED', fp16_transform)])
    else:
        dims.append([("FP16=DISABLED", noop_transform)])

    names = [[x[0] for x in dim] for dim in dims]
    funcs = [[x[1] for x in dim] for dim in dims]
    
    all_funcs = itertools.product(*funcs)
    all_names = itertools.product(*names)

    for func_list, name_list in zip(all_funcs, all_names):
        yield (func_list, name_list)


def load_optimization_config(model_path, olive_config_dict):
    opt_config = OptimizationConfig(
        model_path=model_path,
        **olive_config_dict
    )
    return opt_config

def main(model_path, output_dir, rewrite_config, olive_config):

    logging.info(f"Loaded rewrite config: {rewrite_config}")
    logging.info(f"Loaded olive config: {olive_config}")

    for func_list, name_list in get_configurations(rewrite_config):

        config_name = '|'.join(name_list) if name_list else 'DEFAULT=0'
        logger.info("Running configuration: " + config_name)

        out_dir = os.path.join(output_dir, config_name)        
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

        with open(model_path, 'rb') as fh:
            onnx_model = onnx.load_from_string(fh.read())
        
        for name, func in zip(name_list, func_list):
            onnx_model = func(onnx_model)

        converted_model_path = os.path.join(out_dir, 'converted_model.onnx')
        save_model(onnx_model, converted_model_path)

        olive_config['result_path'] = out_dir
        opt_config = load_optimization_config(converted_model_path, olive_config)
        optimize(opt_config)

    logging.info(f"Run complete; writing summarized results to {output_path}")
#    write_csv_summary(output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config.json file for optimization", required=True)
    parser.add_argument("-i", "--input", help="path to input model", dest="_input")
    parser.add_argument("-o", "--output", help="path to store final output locally")
    parser.add_argument("-g", "--gpu", help="Use GPU", action='store_true')

    args = parser.parse_args()

    config_path = args.config

    with smart_open.open(config_path) as fh:
        config_dict = json.load(fh)

        if args.gpu:
            olive_config = config_dict['olive_config_cuda']
            rewrite_config = config_dict['rewrite_config_cuda']
        else:
            olive_config = config_dict['olive_config_cpu']
            rewrite_config = config_dict['rewrite_config_cpu']

        model_path = config_dict.get("model_path", None) or args._input

    model_name = model_path.split('/')[-1]
    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    output_path = args.output or f"/tmp/{model_name}_{timestamp}"

    assert model_path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    local_model_copy = os.path.join(output_path, 'input.onnx')

    with smart_open.open(model_path, 'rb') as fh:
        logging.info(f"Fetching local copy: {model_path} => {local_model_copy}")

        model = onnx.load(fh)
        if 'input_sizes' in config_dict:            
            for input_name, input_shape in config_dict['input_sizes'].items():
                logging.info(f"Setting {input_name} to shape {input_shape}")
                onnx_model_utils.make_input_shape_fixed(model.graph, input_name, input_shape)

        onnx.save(model, local_model_copy)

    main(local_model_copy, output_path, rewrite_config, olive_config)



