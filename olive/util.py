from time import perf_counter_ns
import numpy as np


def generate_npz_files(output_npz_path, name_list, value_list):
    assert name_list
    assert len(name_list) == len(value_list)
    data = dict(zip(name_list, value_list))
    np.savez(output_npz_path, **data)


def is_npz_format(numpy_file_path):
    with np.load(numpy_file_path, allow_pickle=True) as data:
        return isinstance(data, np.lib.npyio.NpzFile)


def load_npz_file(file_path):
    with np.load(file_path, allow_pickle=True) as npz_data:
        return dict(npz_data.items())


def reorder_npz_data(npz_data, input_names=None):
    if input_names:
        test_data = []
        for i in input_names:
            test_data.append(npz_data.get(i))
    else:
        test_data = list(npz_data.values())
    return test_data


def if_enumerable(data):
    try:
        for _ in data:
            pass
        return True
    except TypeError as e:
        return False


def benchmark(
    fn,
    warmup_count: int,
    benchmark_count: int,
):
    print(f"Warmup {warmup_count} run(s).")
    for _ in range(warmup_count):
        fn()

    print(f"Benchmark {benchmark_count} run(s).")
    start_ns = perf_counter_ns()
    for _ in range(benchmark_count):
        fn()
    elapsed_ns = perf_counter_ns() - start_ns
    average_ms = elapsed_ns / benchmark_count / 1e6
    return (average_ms, benchmark_count)        
