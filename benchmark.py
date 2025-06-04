#!/usr/bin/env python3
# 
# Benchmark the performance of nixl with/without ray

import logging
import threading
import time
import ray
from tabulate import tabulate
from pydantic import BaseModel

from nixl_example_no_ray import Target, Initiator
from nixl_example_ray import TargetActor, InitiatorActor

logger = logging.getLogger(__name__)

BUF_SIZE = 32768 * 15424

def bm_nixl_no_ray(*, buf_size: int = BUF_SIZE, concurrency: int = 1, num_trials: int = 1) -> tuple[float, float]:
    """
    Benchmark the performance of nixl without ray.

    Args:
        buf_size: The size of the buffer to use for the transfer.
        concurrency: The number of initiators that concurrently read from the target.
        num_trials: The number of trials to run for each initiator.

    Returns:
        tuple[float, float]: The time taken for the benchmark (in seconds) and the bandwidth in GB/s.
    """
    print("--" * 20 + " Benchmark begins " + "--" * 20)
    target = Target(buf_size=buf_size)
    initiators = [Initiator(name=f"initiator_{i}", buf_size=buf_size) for i in range(concurrency)]
    target_meta = target.prepare_for_read()

    tags: dict[str, list[bytes]] = {}
    initiator_threads = []
    for initiator in initiators:
        _tags = [f"trial_{j}" for j in range(num_trials)]
        tags[initiator.name] = [tag.encode() for tag in _tags]

        def initiator_thread():
            for tag in _tags:
                initiator.blocking_read_remote(target_meta, tag=tag)
        initiator_threads.append(threading.Thread(target=initiator_thread))

    def target_thread():
        for initiator in initiators:
            for tag in tags[initiator.name]:
                target.blocking_wait_for_read(initiator.name, tag=tag)

    target_thread = threading.Thread(target=target_thread)

    begin_time = time.perf_counter()
    for initiator_thread in initiator_threads:
        initiator_thread.start()
    target_thread.start()

    for initiator_thread in initiator_threads:
        initiator_thread.join()
    target_thread.join()
    duration = time.perf_counter() - begin_time

    total_MB = buf_size * num_trials * concurrency / 1024 / 1024
    bandwidth = total_MB / duration / 1024
    print(f"Benchmark completed in {duration:.2f} seconds, {bandwidth:.3f} GB/s. {num_trials} trials, {concurrency} readers.")

    target.cleanup()
    for initiator in initiators:
        initiator.cleanup()

    return duration, bandwidth

def bm_nixl_ray(*, buf_size: int = BUF_SIZE, concurrency: int = 1, num_trials: int = 1) -> tuple[float, float]:
    """
    Benchmark the performance of nixl with ray.

    Args:
        buf_size: The size of the buffer to use for the transfer.
        concurrency: The number of initiators that concurrently read from the target.
        num_trials: The number of trials to run for each initiator.

    Returns:
        tuple[float, float]: The time taken for the benchmark (in seconds) and the bandwidth in GB/s.
    """
    print("--" * 20 + " Benchmark begins " + "--" * 20)
    if not ray.is_initialized():
        ray.init()

    remote_options = dict(
        runtime_env=dict(env_vars={}),
        num_gpus=1,
    )

    target = TargetActor.options(**remote_options).remote(buf_size=buf_size)
    initiator_names = [f"initiator_{i}" for i in range(concurrency)]
    initiators = [InitiatorActor.options(**remote_options).remote(name=name, buf_size=buf_size) for name in initiator_names]
    target_meta = target.prepare_for_read.remote()

    tags: dict[str, list[bytes]] = {}
    initiator_handles = []
    for initiator, name in zip(initiators, initiator_names):
        _tags = [f"trial_{j}".encode() for j in range(num_trials)]
        tags[name] = _tags
        initiator_handles.extend([initiator.blocking_read_remote.remote(target_meta, tag=tag) for tag in _tags])

    target_handles = []
    for name in initiator_names:
        for tag in tags[name]:
            target_handles.append(target.blocking_wait_for_read.remote(name, tag=tag))

    begin_time = time.perf_counter()
    
    ray.get(initiator_handles)
    ray.get(target_handles)

    duration = time.perf_counter() - begin_time

    total_MB = buf_size * num_trials * concurrency / 1024 / 1024
    bandwidth = total_MB / duration / 1024
    print(f"Benchmark completed in {duration:.2f} seconds, {bandwidth:.3f} GB/s. {num_trials} trials, {concurrency} readers.")

    ray.get(target.cleanup.remote())
    for initiator in initiators:
        ray.get(initiator.cleanup.remote())

    return duration, bandwidth


class BenchmarkResult(BaseModel):
    use_ray: str
    duration: float
    bandwidth: float
    concurrency: int
    num_trials: int


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    ray.init(logging_level="error")

    results: list[BenchmarkResult] = []

    for (concurrency, num_trials) in [(1, 1), (1, 16), (8, 16)]:
        duration, bandwidth = bm_nixl_no_ray(concurrency=concurrency, num_trials=num_trials)
        results.append(BenchmarkResult(use_ray="N", duration=duration, bandwidth=bandwidth, concurrency=concurrency, num_trials=num_trials))
        duration, bandwidth = bm_nixl_ray(concurrency=concurrency, num_trials=num_trials)
        results.append(BenchmarkResult(use_ray="Y", duration=duration, bandwidth=bandwidth, concurrency=concurrency, num_trials=num_trials))

    # Print in sheet format
    print(tabulate([results.model_dump() for results in results], headers="keys", tablefmt="grid"))
