run_pytorch_profiler.py
==============================

Script to check performance of key steps in the docking models using pytorch's built-in profiler.
This can be used to optimize models by identifying performance bottlenecks or operations to avoid overusing.

Currently, the script loads an experiment and trains for 1 epoch as warmup, then 1 epoch using the profiler.
The profiler will produced a table showing resource usage for specific operations such as time, cpu percentage, etc.
