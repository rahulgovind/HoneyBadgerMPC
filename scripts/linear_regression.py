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


async def _linear_regression_mpc_program(ctx, X, y):
    """
    Given data y = 9 * x_ 1 + 4 * x_2 + 7 * x_3 + 2
    Find the coefficients of the best fit line
    Should be (9, 4, 7, 2)
    """
    y = y.reshape(-1, 1)

    x = FixedPointNDArray(ctx, X)
    y = FixedPointNDArray(ctx, y)

    X_normalized, x_mean, x_std = await normalize(x)
    y_normalized, y_mean, y_std = await normalize(y, axis=None)

    print(await y_normalized.open())
    theta = await linear_regression_mpc(ctx, X_normalized, y_normalized,
                                        epochs=40,
                                        learning_rate=0.25)
    await theta.resolve()
    # theta = await theta.open()
    ndim = X.shape[1]

    m = theta[:ndim].reshape(ndim) / x_std * y_std
    await m.resolve()

    c = (theta[ndim] * y_std - (x_mean.reshape(ndim) * m).sum()) + y_mean
    print("m = ", await m.open())
    print("c = ", await c.open())
    # print(np.std(X, axis=0))
    #
    # print(theta[:3].flatten() / np.std(X, axis=0).flatten()[:3] * y_std_)


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
                X = uniform_random(0, 10, (32, 30), seed=0)
                c = uniform_random(0, 10, (30,), seed=1)
                y = np.sum(c * X, axis=1).flatten()
                # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
                program_runner.add(0, _linear_regression_mpc_program, X=X, y=y)

                await program_runner.join()
                await program_runner.close()


            loop.run_until_complete(_process_prog(
                HbmpcConfig.peers, HbmpcConfig.N, HbmpcConfig.t,
                HbmpcConfig.my_id))
        else:
            program_runner = TaskProgramRunner(n, t, config)
            X = np.random.uniform(0, 10, (15, 3))
            y = 9 * X[:, 0] + 4 * X[:, 1] + 7 * X[:, 2] + 50
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

            program_runner.add(_linear_regression_mpc_program, X=X, y=y)
            loop.run_until_complete(program_runner.join())
    finally:
        loop.close()

    end_time = time.time()
    print(end_time - start_time)
