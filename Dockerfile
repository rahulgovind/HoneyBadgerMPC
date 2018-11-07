FROM quay.io/pypa/manylinux1_x86_64 as wheel_builder

ENV PYTHONUNBUFFERED=1

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain nightly-2018-10-24
ENV PATH /root/.cargo/bin:$PATH
ENV PYBIN /opt/python/cp37-cp37m/bin
ENV PYTHON_SYS_EXECUTABLE "$PYBIN/python"
RUN "${PYBIN}/pip" install -U pip setuptools wheel setuptools-rust

RUN mkdir -p /usr/src/pairing
COPY ./pairing /usr/src/pairing
WORKDIR /usr/src/pairing

RUN "${PYBIN}/python" setup.py bdist_wheel

RUN sh scripts/repair-wheel.sh


FROM python:3.7.1-stretch

ENV PYTHONUNBUFFERED=1

COPY --from=wheel_builder /usr/src/pairing/wheelhouse/ /usr/src/wheelhouse

# TODO try removing ... probably not needed.
COPY --from=wheel_builder /usr/src/pairing/scripts/_manylinux.py /usr/local/bin/

RUN apt-get update && apt-get install -y vim tmux

RUN apt-get update && apt-get install -y libgmp-dev libmpc-dev libmpfr-dev libntl-dev libflint-dev

RUN pip install --upgrade pip
RUN pip install /usr/src/wheelhouse/*.whl

WORKDIR /usr/src/HoneyBadgerMPC
COPY . /usr/src/HoneyBadgerMPC

# This is needed otherwise the build for the power sum solver will fail.
# This is a known issue in the version of libflint-dev in apt.
# https://github.com/wbhart/flint2/issues/217
# This has been fixed if we pull the latest code from the repo. However, we want
# to avoid compiling the lib from the source since it adds 20 minutes to the build.
RUN sed -i '30c #include "flint/flint.h"' /usr/include/flint/flintxx/flint_classes.h

ARG BUILD
RUN pip install --no-cache-dir -e .[$BUILD]

RUN make -C apps/shuffle/cpp
