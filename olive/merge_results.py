import logging
import os
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def write_csv_summary(result_dir):
    all_dfs = []

    for sub_dir in os.listdir(result_dir):
        result_path = os.path.join(result_dir, sub_dir, 'olive_result.csv')
        if not os.path.exists(result_path):
            logger.info("Skipping non-existing result from: " + result_path)
            continue

        logger.info("Reading results from: " + result_path)
        with open(result_path) as fh:
            df = pd.read_csv(fh)
            num_columns = len(df.columns)
            for idx, dimension in enumerate(sub_dir.split('|')):
                
                dim_name, dim_value = dimension.split('=')
                df.insert(num_columns + idx, dim_name, dim_value)
                all_dfs.append(df)

    combined_csv = pd.concat(all_dfs)
    out_file = os.path.join(result_dir, 'merged.csv')

    logger.info("Writing results to: " + out_file)
    combined_csv.to_csv(out_file, header=True, index=False,  encoding='utf-8')

if __name__ == '__main__':
    write_csv_summary(sys.argv[1])
