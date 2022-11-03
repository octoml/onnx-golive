import logging
import datetime
import os
import sys
import pandas as pd

from olive.instance_type import get_instance_type

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_model_name(dir_name):
    toks = dir_name.split('/')
    last_tok = toks[-1] or toks[-2]
    return last_tok.split('.')[0]

def merge_model_outputs(model_dir):
    all_dfs = []

    print(model_dir)
    model_name = get_model_name(model_dir)
    logging.info(f"Loading results for {model_name} from {model_dir}")

    for sub_dir in os.listdir(model_dir):
        result_path = os.path.join(model_dir, sub_dir, 'olive_result.csv')
        if not os.path.exists(result_path):
            logger.info("Skipping non-existing result from: " + result_path)
            continue

        with open(result_path) as fh:
            df = pd.read_csv(fh)
            num_columns = len(df.columns)
            for idx, dimension in enumerate(sub_dir.split('|')):
                dim_name, dim_value = dimension.split('=')
                df.insert(num_columns + idx, dim_name, dim_value)
                all_dfs.append(df)

    combined_df = pd.concat(all_dfs)
    combined_df.insert(0, "model_name", model_name)    
    return combined_df

def write_csv_summary(result_dirs):
    instance_type = get_instance_type()
    assert instance_type

    print(f"Loading models on instance {instance_type};  {list(sys.argv[1:])}")

    model_results = (merge_model_outputs(_dir) for _dir in result_dirs)
    all_model_results = pd.concat(model_results)
    all_model_results.insert(0, "instance_type", instance_type)

    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    out_file = os.path.join(os.getcwd(), f"out_{timestamp}.csv")

    logging.info("Merging results to " + out_file)
    all_model_results.to_csv(out_file, header=True, index=False, encoding='utf-8')

if __name__ == '__main__':
    write_csv_summary(sys.argv[1:])
