import itertools
import logging
import os
import re
import time
from copy import copy
from multiprocessing import Barrier, Manager, Process

import numpy as np
import onnxruntime as ort
from packaging import version

from ..constants import ONNX_TO_NP_TYPE_MAP, SUB_PROCESS_NAME_PREFIX
from .mlperf_dataset import Dataset
from .server_runner import ServerRunner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ort.set_default_logger_severity(3)

PRETUNING_SESSION_NAME = "pretuning"


def tune_onnx_model(optimization_config):
    regenerate_input_data(optimization_config)
    all_test_results = []
    for tuning_combo in generate_tuning_combos(optimization_config):
        all_test_results.extend(threads_num_tuning(optimization_config, tuning_combo))
    return all_test_results


def regenerate_input_data(optimization_config):
    input_dict = {}
    execution_provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPUExecutionProvider"
    inputs = ort.InferenceSession(optimization_config.model_path, providers=[execution_provider]).get_inputs()
    for i in range(0, len(inputs)):
        if inputs[i].type in ONNX_TO_NP_TYPE_MAP.keys():
            input_dict[inputs[i].name] = optimization_config.inference_input_dict.get(inputs[i].name).astype(ONNX_TO_NP_TYPE_MAP[inputs[i].type])
        else:
            raise KeyError("failed in mapping operator {} which has type {}".format(
                inputs[i].name, inputs[i].type))
    optimization_config.inference_input_dict = input_dict


def generate_tuning_combos(optimization_config):
    if optimization_config.openmp_enabled or version.parse(ort.__version__) <= version.parse('1.7.0'):
        tuning_combos = itertools.product(optimization_config.omp_wait_policy_list, optimization_config.kmp_affinity,
                                          optimization_config.omp_max_active_levels, optimization_config.providers_list,
                                          optimization_config.execution_mode_list, optimization_config.ort_opt_level_list, optimization_config.concurrency_num_list)
    else:
        tuning_combos = itertools.product([None], [None], [None],
                                          optimization_config.providers_list, optimization_config.execution_mode_list,
                                          optimization_config.ort_opt_level_list, optimization_config.concurrency_num_list)
    yield from tuning_combos


def threads_num_tuning(optimization_config, tuning_combo):
    cpu_cores = optimization_config.cpu_cores
    tuning_results = []

    if tuning_combo[0]:
        os.environ["OMP_WAIT_POLICY"] = tuning_combo[0]
    if tuning_combo[1]:
        os.environ["KMP_AFFINITY"] = tuning_combo[1]
    if tuning_combo[2]:
        os.environ["OMP_MAX_ACTIVE_LEVELS"] = tuning_combo[2]
    provider = tuning_combo[3]
    execution_mode = tuning_combo[4]
    ort_opt_level = tuning_combo[5]
    concurrency_num = tuning_combo[6]

    test_params = dict()
    test_params["execution_provider"] = provider
    test_params["execution_mode"] = execution_mode
    test_params["graph_optimization_level"] = ort_opt_level
    test_params['concurrency_num'] = concurrency_num

    if provider == "TensorrtExecutionProvider" and optimization_config.trt_fp16_enabled:
        os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"

    try:
        for inter in optimization_config.inter_thread_num_list:
            test_params["inter_op_num_threads"] = inter
            for intra in optimization_config.intra_thread_num_list:
                test_params["intra_op_num_threads"] = intra
                threads_num_binary_search(optimization_config, test_params, tuning_results, cpu_cores)
    except Exception:
        logger.error("Optimization failed for tuning combo {}".format(tuning_combo))
        pass

    if provider == "TensorrtExecutionProvider" and optimization_config.trt_fp16_enabled and optimization_config.fp32_enabled:
        os.environ["ORT_TENSORRT_FP16_ENABLE"] = "0"
        try:
            for inter in optimization_config.inter_thread_num_list:
                test_params["inter_op_num_threads"] = inter
                for intra in optimization_config.intra_thread_num_list:
                    test_params["intra_op_num_threads"] = intra
                    threads_num_binary_search(optimization_config, test_params, tuning_results, cpu_cores)
        except Exception:
            logger.error("Optimization failed for tuning combo {}".format(tuning_combo))
            pass

    return tuning_results


def threads_num_binary_search(optimization_config, test_params, tuning_results, cpu_cores):
    if test_params.get("execution_mode") == ort.ExecutionMode.ORT_SEQUENTIAL:
        threads_names = ["intra_op_num_threads"]
    else:
        threads_names = ["inter_op_num_threads", "intra_op_num_threads"]
    best_throughput = None
    best_latency = None

    if test_params.get("inter_op_num_threads") and test_params.get("intra_op_num_threads"):
        test_result = get_benchmark(optimization_config, test_params)
        if test_result:
            tuning_results.append(test_result)

        if optimization_config.throughput_tuning_enabled:
            best_throughput = test_result["throughput"]
        else:
            best_latency = test_result["latency_ms"]["avg"]

    else:
        for threads_name in threads_names:
            thread_num = test_params.get(threads_name)
            if thread_num is not None:
                upper_threads_num = thread_num
                lower_threads_num = thread_num
            else:
                upper_threads_num = cpu_cores
                lower_threads_num = 1

            best_threads_num = lower_threads_num
            current_threads_num = lower_threads_num
            test_params[threads_name] = current_threads_num

            test_result = get_benchmark(optimization_config, test_params)
            if test_result:
                tuning_results.append(test_result)

            if optimization_config.throughput_tuning_enabled:
                best_throughput = test_result["throughput"]
            else:
                best_latency = test_result["latency_ms"]["avg"]

            current_threads_num = upper_threads_num

            while lower_threads_num < upper_threads_num:
                test_params[threads_name] = current_threads_num

                test_result = get_benchmark(optimization_config, test_params)
                if test_result:
                    tuning_results.append(test_result)

                mid_threads_num = lower_threads_num + (upper_threads_num - lower_threads_num) // 2
                if (best_throughput and best_throughput > test_result["throughput"]) or (best_latency and best_latency < test_result["latency_ms"]["avg"]):
                    upper_threads_num = mid_threads_num
                    current_threads_num = upper_threads_num
                else:
                    lower_threads_num = mid_threads_num + 1
                    best_threads_num = current_threads_num
                    current_threads_num = lower_threads_num

            test_params[threads_name] = best_threads_num


def generate_test_name(test_params):
    if not test_params:
        return PRETUNING_SESSION_NAME
    test_name = "_".join(["_".join([str(v) for v in i]) for i in test_params.items()])
    for env_var in ["OMP_WAIT_POLICY", "KMP_AFFINITY", "OMP_MAX_ACTIVE_LEVELS"]:
        env_val = os.getenv(env_var)
        if env_val:
            test_name += "_{}_{}".format(env_var, env_val)
    return test_name


def create_inference_session(model_path, test_params=None, available_providers=None):
    sess_options = ort.SessionOptions()
    if version.parse(ort.__version__) >= version.parse("1.11.0"):
        sess_options.add_session_config_entry('session.dynamic_block_base', '4')

    if test_params:
        execution_provider = test_params.get("execution_provider")
        inter_op_num_threads = test_params.get("inter_op_num_threads")
        intra_op_num_threads = test_params.get("intra_op_num_threads")
        execution_mode = test_params.get("execution_mode")
        graph_optimization_level = test_params.get("graph_optimization_level")
        if inter_op_num_threads:
            sess_options.inter_op_num_threads = inter_op_num_threads
        if intra_op_num_threads:
            sess_options.intra_op_num_threads = intra_op_num_threads
            os.environ['OMP_NUM_THREADS'] = str(intra_op_num_threads)
        if execution_mode:
            sess_options.execution_mode = execution_mode
        if graph_optimization_level:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel(graph_optimization_level)
        if execution_provider:
            onnx_session = ort.InferenceSession(model_path, sess_options, providers=[execution_provider])
        else:
            onnx_session = ort.InferenceSession(model_path, sess_options)
    else:
        available_providers = available_providers if available_providers else ort.get_available_providers()
        execution_provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in available_providers else "CPUExecutionProvider"
        logger.info("Using default provider {}".format(execution_provider))
        onnx_session = ort.InferenceSession(model_path, sess_options, providers=[execution_provider])
    return onnx_session


def get_benchmark(optimization_config, test_params={}):
    if optimization_config.throughput_tuning_enabled:
        # throughput tuning uses the main process for results (are they extrapolated?)
        # the other processes are there to simulate load
        manager = Manager()
        test_result = manager.dict()
        main_process = Process(name="main_process", target=get_throughput,
                               args=(optimization_config, test_params, test_result))

        process_list = []
        concurrency_num = test_params.get('concurrency_num', 1)

        num_of_background = concurrency_num - 1
        if num_of_background > 0:
            synchronizer = Barrier(num_of_background + 1)
            for i in range(0, num_of_background):
                p = Process(name="{}_{}".format(SUB_PROCESS_NAME_PREFIX, i), target=get_throughput,
                            args=(optimization_config, test_params, None, synchronizer))
                p.start()
                logger.info("PID [{}] started".format(p.pid))
                process_list.append(p)
            synchronizer.wait()

        # execute main func, we only collect benchmark from this main func
        main_process.start()
        main_process.join()

        # once the main func finished, stop child process
        for p in process_list:
            if p.is_alive():
                p.terminate()
                logger.info("PID {} killed".format(p.pid))
        return dict(test_result)
    else:
        test_result = benchmark_latency(optimization_config, test_params)
        return test_result


def benchmark_latency(optimization_config, test_params):
    # Session name will be "pretuning" if there are no parameters
    session_name = generate_test_name(test_params)

    manager = Manager()
    latencies = manager.list()

    # Do a warm up in the main process to test for an early out
    should_perform_test_run = True
    if test_params and not optimization_config.run_all:
        # Number of tests to determine early out criteria
        EARLYOUT_TEST_COUNT = 20
        # Skip test if latencies are X times more than pretuning
        EARLYOUT_LATENCY_FACTOR = 5

        earlyout_optimization_config = copy(optimization_config)
        earlyout_optimization_config.test_num = EARLYOUT_TEST_COUNT
        earlyout_latencies = []
        get_latency(earlyout_optimization_config, test_params, earlyout_latencies)

        earlyout_latency_ms = sum(earlyout_latencies) / len(earlyout_latencies) * 1000
        if (earlyout_latency_ms > (earlyout_optimization_config.pretuning_latency_ms * EARLYOUT_LATENCY_FACTOR)):
            should_perform_test_run = False
            latencies.extend(earlyout_latencies)
            session_name += "_earlyout"

    # Perform the test run if we haven't eliminated this config via early out testing above
    if should_perform_test_run:
        concurrency_num = test_params.get('concurrency_num', 1)
        process_list = []
        synchronizer = Barrier(concurrency_num+1)

        for i in range(0, concurrency_num):
            p = Process(name="{}_{}".format(SUB_PROCESS_NAME_PREFIX, i), target=get_latency,
                        args=(optimization_config, test_params, latencies, synchronizer))
            p.start()
            logger.info("[{}] process started".format(p.pid))
            process_list.append(p)

        synchronizer.wait()

        for p in process_list:
            p.join()
            logger.info("[{}] process finished".format(p.pid))
            p.terminate()


    # Aggregate latencies to test results
    test_result = dict()
    test_result["test_name"] = session_name
    if test_params:
        test_result["execution_provider"] = test_params.get("execution_provider")
        test_result["env_vars"] = {
            "OMP_WAIT_POLICY": str(os.getenv("OMP_WAIT_POLICY")),
            "OMP_NUM_THREADS": str(os.getenv("OMP_NUM_THREADS")),
            "KMP_AFFINITY": str(os.getenv("KMP_AFFINITY")),
            "OMP_MAX_ACTIVE_LEVELS": str(os.getenv("OMP_MAX_ACTIVE_LEVELS")),
            "ORT_TENSORRT_FP16_ENABLE": str(os.getenv("ORT_TENSORRT_FP16_ENABLE", 0))
        }
        test_result["session_options"] = {
            "inter_op_num_threads": str(test_params.get("inter_op_num_threads")),
            "intra_op_num_threads": str(test_params.get("intra_op_num_threads")),
            "execution_mode": str(test_params.get("execution_mode")),
            "graph_optimization_level": str(test_params.get("graph_optimization_level")),
            "concurrency_num": test_params.get("concurrency_num")
        }

    test_result["latency_ms"] = {
        "avg": round(sum(latencies) / len(latencies) * 1000, 5),
        "latency_p50": round(np.percentile(latencies, 50) * 1000, 5),
        "latency_p75": round(np.percentile(latencies, 75) * 1000, 5),
        "latency_p90": round(np.percentile(latencies, 90) * 1000, 5),
        "latency_p95": round(np.percentile(latencies, 95) * 1000, 5),
        "latency_p99": round(np.percentile(latencies, 99) * 1000, 5),
        "latency_p999": round(np.percentile(latencies, 99.9) * 1000, 5),
    }
    logger.info("ONNX model average inference time = {} for test {}".format(test_result["latency_ms"]["avg"], session_name))
    test_result["throughput"] = 1000 / test_result["latency_ms"]["avg"] if test_result["latency_ms"]["avg"] != 0 else "null"
    return test_result


def get_latency(optimization_config, test_params, latency_results, synchronizer=None):
    onnx_session = create_inference_session(optimization_config.model_path, test_params, optimization_config.providers_list)
    onnx_output_names = optimization_config.output_names if optimization_config.output_names else [o.name for o in onnx_session.get_outputs()]

    # warmup
    for i in range(optimization_config.warmup_num):
        onnx_session.run(onnx_output_names, optimization_config.inference_input_dict)

    if synchronizer:
        synchronizer.wait()

    # run test
    latencies = []
    for _ in range(optimization_config.test_num):
        t = time.perf_counter()
        onnx_session.run(onnx_output_names, optimization_config.inference_input_dict)
        latencies.append(time.perf_counter() - t)

    latency_results.extend(latencies)


def get_throughput(optimization_config, test_params, test_result=None, synchronizer=None):
    onnx_session = create_inference_session(optimization_config.model_path, test_params, optimization_config.providers_list)
    onnx_output_names = optimization_config.output_names if optimization_config.output_names else [o.name for o in onnx_session.get_outputs()]

    # create server runner
    ds = Dataset(onnx_session, optimization_config.inputs_spec)
    runner = ServerRunner(onnx_session, ds, optimization_config, onnx_output_names)

    # warmup
    runner.warmup(optimization_config.warmup_num)

    if synchronizer:
        synchronizer.wait()

    # run test
    runner.start_run()
    runner.finish()

    if test_result:
        session_name = generate_test_name(test_params)
        is_valid, latency_result, throughput_result = parse_mlperf_log(optimization_config.result_path)
        if is_valid:
            test_result["test_name"] = session_name
            if test_params:
                test_result["execution_provider"] = test_params.get("execution_provider")
                test_result["env_vars"] = {
                    "OMP_WAIT_POLICY": str(os.getenv("OMP_WAIT_POLICY")),
                    "OMP_NUM_THREADS": str(os.getenv("OMP_NUM_THREADS")),
                    "KMP_AFFINITY": str(os.getenv("KMP_AFFINITY")),
                    "OMP_MAX_ACTIVE_LEVELS": str(os.getenv("OMP_MAX_ACTIVE_LEVELS")),
                    "ORT_TENSORRT_FP16_ENABLE": str(os.getenv("ORT_TENSORRT_FP16_ENABLE", 0))
                }
                test_result["session_options"] = {
                    "inter_op_num_threads": str(test_params.get("inter_op_num_threads")),
                    "intra_op_num_threads": str(test_params.get("intra_op_num_threads")),
                    "execution_mode": str(test_params.get("execution_mode")),
                    "graph_optimization_level": str(test_params.get("graph_optimization_level"))
                }

            is_valid, latency_result, throughput_result = parse_mlperf_log(optimization_config.result_path)
            test_result["latency_ms"] = latency_result
            test_result["throughput"] = throughput_result
            logger.info("Optimal QPS = {} for test {}".format(test_result["throughput"], session_name))
        else:
            logger.error("Benchmark is not valid for test {}. Please increase the expected latency".format(session_name))


def parse_mlperf_log(result_path):
    is_valid = False
    result = {}
    latency_result = {}
    fname = os.path.join(result_path, "mlperf_log_summary.txt")
    with open(fname, "r") as f:
        for line in f:
            m = re.match(r"^Result\s+is\s*\:\s+VALID", line)
            if m:
                is_valid = True
            m = re.match(r"^\s*([\w\s.\(\)\/]+)\s*\:\s*([\w\+\.]+).*", line)
            if m:
                result[m.group(1).strip()] = m.group(2).strip()
    throughput_result = float(result.get("Scheduled samples per second"))

    def parse_latency_ms(n):
        return float(result[n]) / 1000000

    latency_result["avg"] = parse_latency_ms("Mean latency (ns)")
    latency_result["latency_p50"] = parse_latency_ms("50.00 percentile latency (ns)")
    latency_result["latency_p90"] = parse_latency_ms("90.00 percentile latency (ns)")
    latency_result["latency_p95"] = parse_latency_ms("95.00 percentile latency (ns)")
    latency_result["latency_p99"] = parse_latency_ms("99.00 percentile latency (ns)")
    latency_result["latency_p99.9"] = parse_latency_ms("99.90 percentile latency (ns)")

    return is_valid, latency_result, throughput_result
