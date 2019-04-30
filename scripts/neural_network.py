import logging
import asyncio
import time
from honeybadgermpc.mpc import PreProcessedElements, TaskProgramRunner
from honeybadgermpc.mixins import BeaverTriple, MixinOpName
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import gc
from honeybadgermpc.fixedpoint import FixedPointNDArray, concatenate, uniform_random, \
    set_ppe, sigmoid, sigmoid_deriv


class NeuralNetwork(object):
    def __init__(self, hidden_size, learning_rate, epochs):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs

    async def fit(self, ctx, X, y):
        self.input_size = X.shape[1]
        self.output_size = y.shape[0]

        self.w1 = FixedPointNDArray(ctx, uniform_random(0, 1, (self.input_size,
                                                               self.hidden_size),
                                                        seed=0))
        self.w2 = FixedPointNDArray(ctx, uniform_random(0, 1,
                                                        (self.hidden_size, 1),
                                                        seed=1))

        for epoch in range(self.epochs):
            start_time = time.time()
            # Forward
            print(f"-------------- Starting epoch {epoch} ------------------")
            print(f"-------------- Evaluating input layer ------------------")
            start_time2 = time.time()
            y1 = X @ self.w1
            # print("y1 done")
            await y1.resolve()
            end_time2 = time.time()
            print(f"First layer linear operations took {end_time2 - start_time2}s")

            start_time2 = time.time()
            l1 = await sigmoid(y1)
            await l1.resolve()
            end_time2 = time.time()
            print(f"First layer non-linear ops took {end_time2 - start_time2}s")
            gc.collect()

            print(f"-------------- Evaluating hidden layer -----------------")
            y2 = l1 @ self.w2
            await y2.resolve()

            l2 = await sigmoid(y2)

            await l2.resolve()
            gc.collect()

            print(f"--------------- Starting error evaluation --------------")
            #             print("l2: ", l2)

            err = y - l2
            print("Error: ", np.linalg.norm(await err.open()))

            print(f"--------------- Starting back-propagation ---------------")
            l2_delta = err * sigmoid_deriv(l2)
            await l2_delta.resolve()
            l1_delta = (l2_delta @ self.w2.T) * sigmoid_deriv(l1)
            await l1_delta.open()
            gc.collect()

            self.w2 = self.w2 + (l1.T @ l2_delta) * self.learning_rate
            self.w1 = self.w1 + (X.T @ l1_delta) * self.learning_rate
            await self.w1.resolve()
            await self.w2.resolve()
            print(f"--------------- Back-propagation complete ---------------")

            gc.collect()
            end_time = time.time()
            print(f"Epoch took {end_time - start_time}")

    async def evaluate(self, x):
        l1 = await sigmoid(x @ self.w1)
        l2 = await sigmoid(l1 @ self.w2)
        return (l2 > 0.5)


def accuracy(y_pred, y_actual):
    return np.sum(y_pred.reshape(-1, 1) == y_actual.reshape(-1, 1)) / len(y_actual)


async def _neural_network_mpc_program(ctx, x_train, y_train, x_test, y_test):
    # Currently, normalization is done in a non-MPC fashion. However, this shouldn't be
    # the case since normalization is a global operation.
    nn = NeuralNetwork(12, 0.24, 2)
    x_train = FixedPointNDArray(ctx, x_train)
    y_train = FixedPointNDArray(ctx, y_train.reshape(-1, 1))
    await nn.fit(ctx, x_train, y_train)
    x_test = FixedPointNDArray(ctx, x_test)
    evaluations = await nn.evaluate(x_test)
    print("Accuracy: ", accuracy((await evaluations.open()), y_test))


if __name__ == "__main__":
    n = 5
    t = 1
    multiprocess = True
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
            from honeybadgermpc.config import HbmpcConfig
            from honeybadgermpc.ipc import ProcessProgramRunner


            async def _process_prog(peers, n, t, my_id):
                program_runner = ProcessProgramRunner(peers, n, t, my_id,
                                                      config)
                await program_runner.start()
                df = pd.read_csv('data/data.csv')
                del df['Unnamed: 32']

                X = df.iloc[:, 2:].values
                y = np.vectorize(lambda x: 1 if x == 'M' else 0)(df.iloc[:, 1].values)
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                                    random_state=0)

                # Normalize values
                x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
                x_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)

                program_runner.add(0, _neural_network_mpc_program,
                                   x_train=x_train[:32], y_train=y_train[:32],
                                   x_test=x_test[:32], y_test=y_test[:32])

                await program_runner.join()
                await program_runner.close()


            loop.run_until_complete(_process_prog(
                HbmpcConfig.peers, HbmpcConfig.N, HbmpcConfig.t,
                HbmpcConfig.my_id))
        else:
            program_runner = TaskProgramRunner(n, t, config)
            df = pd.read_csv('data/data.csv')
            del df['Unnamed: 32']

            X = df.iloc[:, 2:].values
            y = np.vectorize(lambda x: 1 if x == 'M' else 0)(df.iloc[:, 1].values)
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                                random_state=0)

            # Normalize values
            x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
            x_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)

            program_runner.add(_neural_network_mpc_program,
                               x_train=x_train[:32], y_train=y_train[:32],
                               x_test=x_test[:32], y_test=y_test[:32])

            loop.run_until_complete(program_runner.join())
    finally:
        loop.close()

    end_time = time.time()
    print(end_time - start_time)
