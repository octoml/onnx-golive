import logging
import os
import sys
import pandas as pd

from olive.instance_type import get_instance_type

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def write_csv_summary(result_dir):
    all_dfs = []
    instance_type = get_instance_type() or "UNKNOWN"

    for sub_dir in os.listdir(result_dir):
        result_path = os.path.join(result_dir, sub_dir, 'olive_result.csv')
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

    combined_csv = pd.concat(all_dfs)
    combined_csv.insert(0, "instance_type", instance_type)

    local_file = os.path.join(result_dir, 'merged.csv')
    combined_csv.to_csv(local_file, header=True, index=False, encoding='utf-8')
#    combined_csv.to_csv(out_phile, header=True, index=False,  encoding='utf-8')

if __name__ == '__main__':
    write_csv_summary(sys.argv[1])
