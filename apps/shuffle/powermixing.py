import random
import asyncio
import uuid
import os
import glob
from time import time
from honeybadgermpc.mpc import TaskProgramRunner, Field
from honeybadgermpc.preprocessing import PreProcessedElements, PreProcessingConstants
from honeybadgermpc.preprocessing import wait_for_preprocessing, preprocessing_done
import logging


async def single_secret_phase1(context, **kwargs):
    k = kwargs['k']
    share_id, power_id = kwargs['share_id'], kwargs['power_id']

    pp_elements = PreProcessedElements()
    powers = pp_elements.get_powers(context, power_id)
    a = pp_elements.get_share(context, share_id)
    b = powers[0]
    assert k == len(powers)
    a_minus_b = await (a - b).open()  # noqa: W606
    file_name = f"{share_id}-{context.myid}.input"
    file_path = f"{PreProcessingConstants.SHARED_DATA_DIR}{file_name}"
    with open(file_path, "w") as f:
        print(Field.modulus, file=f)
        print(a.v.value, file=f)
        print(a_minus_b.value, file=f)
        print(k, file=f)
        for power in powers:
            print(power.v.value, file=f)


async def all_secrets_phase1(context, **kwargs):
    k, share_ids = kwargs['k'], kwargs['share_ids']
    power_ids = kwargs['power_ids']
    as_, a_minus_b_shares, all_powers = [], [], []

    pp_elements = PreProcessedElements()

    stime = time()
    for i in range(k):
        a = pp_elements.get_share(context, share_ids[i])
        powers = pp_elements.get_powers(context, power_ids[i])
        a_minus_b_shares.append(a - powers[0])
        as_.append(a)
        all_powers.append(powers)
    bench_logger.info(f"[Phase1] Read shares from file: {time() - stime}")

    stime = time()
    opened_shares = await context.ShareArray(a_minus_b_shares).open()
    bench_logger.info(
        f"[Phase1] Open [{len(a_minus_b_shares)}] a-b shares: {time() - stime}")

    stime = time()
    for i in range(k):
        file_name = f"{share_ids[i]}-{context.myid}.input"
        file_path = f"{PreProcessingConstants.SHARED_DATA_DIR}{file_name}"
        with open(file_path, "w") as f:
            print(Field.modulus, file=f)
            print(as_[i].v.value, file=f)
            print(opened_shares[i].value, file=f)
            print(k, file=f)
            for power in all_powers[i]:
                print(power.v.value, file=f)
    bench_logger.info(f"[Phase1] Write shares to file: {time() - stime}")


async def phase2(node_id, run_id, share_id):
    input_file_name = f"{share_id}-{node_id}.input"
    input_file_path = f"{PreProcessingConstants.SHARED_DATA_DIR}{input_file_name}"
    sum_file_name = f"power-{run_id}_{node_id}.sums"
    sum_file_path = f"{PreProcessingConstants.SHARED_DATA_DIR}{sum_file_name}"

    # NOTE The binary `compute-power-sums` is generated via the command
    # make -C apps/shuffle/cpp
    # and is stored under /usr/local/bin/
    runcmd = f"compute-power-sums {input_file_path} {sum_file_path}"
    await run_command_sync(runcmd)


async def run_command_sync(command):
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()

    logging.debug(f"Command:{command}")
    logging.debug(f"Output: {stdout}")
    if len(stderr):
        logging.info(f"Error: {stderr}")


async def prepare_one_input(context, **kwargs):
    k = kwargs['k']
    power_id, share_id = kwargs['power_id'], kwargs['share_id']
    run_id = kwargs['run_id']

    await single_secret_phase1(
        context,
        k=k,
        share_id=share_id,
        power_id=power_id,
        )
    logging.info(f"[{context.myid}] Input prepared for C++ phase.")
    await phase2(context.myid, run_id, share_id)
    logging.info(f"[{context.myid}] C++ phase completed.")


async def phase3(context, **kwargs):
    k, run_id = kwargs['k'], kwargs['run_id']
    sum_file_name = f"power-{run_id}_{context.myid}.sums"
    sum_file_path = f"{PreProcessingConstants.SHARED_DATA_DIR}{sum_file_name}"
    sum_shares = []

    bench_logger = logging.LoggerAdapter(
        logging.getLogger("benchmark_logger"), {"node_id": context.myid})

    stime = time()
    with open(sum_file_path, "r") as f:
        assert Field.modulus == int(f.readline())
        assert k == int(f.readline())
        sum_shares = [context.Share(int(s)) for s in f.read().splitlines()[:k]]
        assert len(sum_shares) == k
    bench_logger.info(f"[Phase3] Read shares from file: {time() - stime}")

    stime = time()
    opened_shares = await context.ShareArray(sum_shares).open()
    bench_logger.info(f"[Phase3] Open [{len(sum_shares)}] shares: {time() - stime}")
    return opened_shares


async def async_mixing(a_s, n, t, k):
    from .solver.solver import solve

    pr1 = TaskProgramRunner(n, t)

    pp_elements = PreProcessedElements()
    run_id = uuid.uuid4().hex
    for a in a_s:
        power_id = pp_elements.generate_powers(k, n, t)
        share_id = pp_elements.generate_share(a, n, t)
        pr1.add(
            prepare_one_input,
            k=k,
            run_id=run_id,
            power_id=power_id,
            share_id=share_id
        )
    await pr1.join()
    pr2 = TaskProgramRunner(n, t)
    pr2.add(phase3, k=k, run_id=run_id)
    powerSums = (await pr2.join())[0]
    logging.info("Shares from C++ phase opened.")
    result = solve([s.value for s in powerSums])
    logging.info("Equation solver completed.")
    return result


async def build_newton_solver():
    await run_command_sync(f"python apps/shuffle/solver/solver_build.py")


async def build_powermixing_cpp_code():
    await run_command_sync(f"make -C apps/shuffle/cpp")


def async_mixing_in_tasks():
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    n, t, k = 3, 1, 2
    a_s = [Field(random.randint(0, Field.modulus-1)) for _ in range(k)]
    try:
        loop.run_until_complete(build_newton_solver())
        logging.info("Solver built.")
        loop.run_until_complete(build_powermixing_cpp_code())
        logging.info("C++ code built.")
        loop.run_until_complete(async_mixing(a_s, n, t, k))
    finally:
        loop.close()


async def async_mixing_in_processes(
            network_info, n, t, k, run_id, node_id, power_ids, share_ids
        ):
    from .solver.solver import solve
    from honeybadgermpc.ipc import ProcessProgramRunner
    from honeybadgermpc.task_pool import TaskPool

    programRunner = ProcessProgramRunner(network_info, n, t, node_id)
    await programRunner.start()
    programRunner.add(
        0,
        all_secrets_phase1,
        k=k,
        power_ids=power_ids,
        share_ids=share_ids
    )
    await programRunner.join()

    pool = TaskPool(256)
    stime = time()
    for i in range(k):
        pool.submit(phase2(node_id, run_id, share_ids[i]))
    await pool.close()
    bench_logger.info(f"[Phase2] Execute CPP code for all secrets: {time() - stime}")

    programRunner.add(1, phase3, k=k, run_id=run_id)
    powerSums = (await programRunner.join())[0]
    await programRunner.close()

    logging.info("Shares from C++ phase opened.")
    stime = time()
    result = solve([s.value for s in powerSums])
    bench_logger.info(f"[SolverPhase] Run Newton Solver: {time() - stime}")
    logging.info("Equation solver completed.")
    logging.debug(result)
    return result


if __name__ == "__main__":
    import sys
    from honeybadgermpc.config import load_config
    from honeybadgermpc.ipc import NodeDetails
    from honeybadgermpc.exceptions import ConfigurationError

    configfile = os.environ.get('HBMPC_CONFIG')
    node_id = os.environ.get('HBMPC_NODE_ID')
    runid = os.environ.get('HBMPC_RUN_ID')

    # override configfile if passed to command
    try:
        node_id = sys.argv[1]
        configfile = sys.argv[2]
        runid = sys.argv[3]
    except IndexError:
        pass

    if not node_id:
        raise ConfigurationError('Environment variable `HBMPC_NODE_ID` must be set'
                                 ' or a node id must be given as first argument.')

    if not configfile:
        raise ConfigurationError('Environment variable `HBMPC_CONFIG` must be set'
                                 ' or a config file must be given as second argument.')

    if not runid:
        raise ConfigurationError('Environment variable `HBMPC_RUN_ID` must be set'
                                 ' or a config file must be given as third argument.')

    config_dict = load_config(configfile)
    node_id = int(node_id)
    N = config_dict['N']
    t = config_dict['t']
    k = config_dict['k']

    bench_logger = logging.LoggerAdapter(
        logging.getLogger("benchmark_logger"), {"node_id": node_id})

    network_info = {
        int(peerid): NodeDetails(addrinfo.split(':')[0], int(addrinfo.split(':')[1]))
        for peerid, addrinfo in config_dict['peers'].items()
    }

    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()

    def handle_async_exception(loop, ctx):
        logging.info('handle_async_exception:')
        if 'exception' in ctx:
            logging.info(f"exc: {repr(ctx['exception'])}")
        else:
            logging.info(f'ctx: {ctx}')
        logging.info(f"msg: {ctx['message']}")

    loop.set_exception_handler(handle_async_exception)
    loop.set_debug(True)

    # Cleanup pre existing sums file
    sums_file = glob.glob(f'{PreProcessingConstants.SHARED_DATA_DIR}*.sums')
    for f in sums_file:
        os.remove(f)

    try:
        if not config_dict['skipPreprocessing']:
            # Need to keep these fixed when running on processes.
            k = config_dict['k']
            assert k < 1000
            a_s = [Field(i) for i in range(1000+k, 1000, -1)]

            file_name = f"{PreProcessingConstants.SHARED_DATA_DIR}shared.state"

            power_ids = []
            share_ids = []
            pp_elements = PreProcessedElements()
            if node_id == 0:
                for i in range(k):
                    power_ids.append(pp_elements.generate_powers(k, N, t))
                    share_ids.append(pp_elements.generate_share(a_s[i], N, t))
                with open(file_name, "w") as f:
                    f.write(",".join(power_ids))
                    f.write("\n")
                    f.write(",".join(share_ids))
                preprocessing_done()
            else:
                loop.run_until_complete(wait_for_preprocessing())
                with open(file_name, "r") as f:
                    power_ids = f.readline().strip().split(",")
                    share_ids = f.readline().strip().split(",")

        loop.run_until_complete(
            async_mixing_in_processes(
                network_info,
                N,
                t,
                k,
                runid,
                node_id,
                power_ids,
                share_ids
            )
        )
    finally:
        loop.close()

    # asynchronusMixingInTasks()
