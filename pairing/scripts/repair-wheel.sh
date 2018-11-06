#!/bin/bash
set -ex

for whl in dist/*.whl; do
    auditwheel repair "$whl"
done
