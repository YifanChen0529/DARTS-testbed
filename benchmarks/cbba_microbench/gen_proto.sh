#!/usr/bin/env bash
# Must run from the repository root inside the DARTS_nano conda environment.
# Generates cbba_pb2.py and cbba_pb2_grpc.py next to cbba.proto.
set -euo pipefail
python3 -m grpc_tools.protoc -I benchmarks/cbba_microbench --python_out=benchmarks/cbba_microbench --grpc_python_out=benchmarks/cbba_microbench benchmarks/cbba_microbench/cbba.proto
