#!/bin/bash

python3 -m cProfile -s tottime profiler.py | tee profiler_20180808.txt
