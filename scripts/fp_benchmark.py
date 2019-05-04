import logging
import asyncio
import traceback
import time
from honeybadgermpc.mpc import PreProcessedElements, TaskProgramRunner
from honeybadgermpc.mixins import BeaverTriple, MixinOpName
from honeybadgermpc.elliptic_curve import Subgroup
from honeybadgermpc.field import GF, GFElement
import random
import numpy as np
import threading
from sklearn.model_selection import train_test_split
import pandas as pd
import gc
from honeybadgermpc.fixedpoint import FixedPointNDArray, concatenate, uniform_random, \
    set_ppe, to_fixed_point_repr


async def linear_regression_mpc(ctx, X, y, epochs=1, learning_rate=0.05):
    X = concatenate(ctx, [X, np.ones(X.shape[0]).reshape(-1, 1)], axis=1)
    theta = FixedPointNDArray(ctx,
                              uniform_random(-1, 1, X.shape[1], seed=0).reshape(-1, 1))

    N = X.shape[0]
    learning_rate = learning_rate / N
    for epoch in range(epochs):
        print(f"Starting epoch {epoch}")
        dtheta = (X.T @ ((X @ theta) - y))
        await dtheta.resolve()
        # print(f"dtheta resolve")
        theta = theta - learning_rate * dtheta
        await theta.resolve()
        err = X @ theta - y
        print("Error: ", np.linalg.norm(await err.open()))
        # print(f"EPOCH {epoch}", "theta: ", (await theta.open()).flatten())
    return theta


async def normalize(x, axis=0):
    mean = x.mean(axis=axis)
    await mean.resolve()

    var = x.var(axis=axis)
    std = var.sqrt()
    await std.resolve()

    res = (x - mean) / std
    await res.resolve()
    return res, mean, std


async def _benchmark_program(ctx):
    total_time = 0.
    for i in range(10):
        x = FixedPointNDArray(ctx, uniform_random(0, 1, (1000,)))
        start_time = time.time()
        await x.open()
        end_time = time.time()
        total_time += end_time - start_time
    total_time /= 10
    print("Open: ", total_time)

    total_time = 0.
    for i in range(10):
        x = FixedPointNDArray(ctx, uniform_random(0, 1, (1000,)))
        start_time = time.time()
        await x.open()
        end_time = time.time()
        total_time += end_time - start_time
    total_time /= 10
    print("Open: ", total_time)

    total_time = 0.
    for i in range(10):
        x = FixedPointNDArray(ctx, uniform_random(0, 1, (1000,)))
        y = FixedPointNDArray(ctx, uniform_random(0, 1, (1000,)))
        start_time = time.time()
        z = (x * y)
        await z.resolve()
        end_time = time.time()
        total_time += end_time - start_time
    total_time /= 10
    print("FPMul: ", total_time)

    total_time = 0.
    for i in range(1):
        x = FixedPointNDArray(ctx, uniform_random(0, 1, (1000,)))
        y = FixedPointNDArray(ctx, uniform_random(0, 1, (1000,)))
        start_time = time.time()
        z = x / y
        await z.resolve()
        end_time = time.time()
        total_time += end_time - start_time
    total_time /= 1
    print("FPDiv: ", total_time)

    total_time = 0.
    for i in range(10):
        x = FixedPointNDArray(ctx, uniform_random(0, 1, (1000,)))
        start_time = time.time()
        z = x.ltz()
        await z.resolve()
        end_time = time.time()
        total_time += end_time - start_time
    total_time /= 10
    print("LTZ: ", total_time)

    total_time = 0.
    for i in range(1):
        x = FixedPointNDArray(ctx, uniform_random(0, 1, (100,)))
        start_time = time.time()
        z = x.sqrt()
        await z.resolve()
        end_time = time.time()
        total_time += end_time - start_time
    total_time /= 1
    print("Sqrt: ", total_time)


if __name__ == "__main__":
    from honeybadgermpc.config import HbmpcConfig

    multiprocess = HbmpcConfig.N is not None
    if HbmpcConfig.N is not None:
        n = HbmpcConfig.N
        t = HbmpcConfig.t
    else:
        n = 5
        t = 1

    _ppe = PreProcessedElements()
    set_ppe(_ppe)
    logging.info("Generating zeros in sharedata/")
    _ppe.generate_zeros(1000, n, t)
    logging.info("Generating random shares of bits in sharedata/")
    _ppe.generate_random_bits(1000, n, t)
    logging.info('Generating random shares in sharedata/')
    _ppe.generate_rands(1000, n, t)
    logging.info('Generating random shares of triples in sharedata/')
    _ppe.generate_triples(1000, n, t)
    logging.info("Generating random doubles in sharedata/")
    _ppe.generate_double_shares(1000, n, t)

    # logging.info('Generating random shares of bits in sharedata/')
    # ppe.generate_bits(1000, n, t)

    start_time = time.time()
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    try:
        config = {MixinOpName.MultiplyShare: BeaverTriple.multiply_shares}
        if multiprocess:
            from honeybadgermpc.ipc import ProcessProgramRunner


            async def _process_prog(peers, n, t, my_id):
                program_runner = ProcessProgramRunner(peers, n, t, my_id,
                                                      config)
                await program_runner.start()
                program_runner.add(0, _benchmark_program)

                await program_runner.join()
                await program_runner.close()


            loop.run_until_complete(_process_prog(
                HbmpcConfig.peers, HbmpcConfig.N, HbmpcConfig.t,
                HbmpcConfig.my_id))
        else:
            program_runner = TaskProgramRunner(n, t, config)
            program_runner.add(_benchmark_program)
            loop.run_until_complete(program_runner.join())
    finally:
        loop.close()

    end_time = time.time()
    print(end_time - start_time)
