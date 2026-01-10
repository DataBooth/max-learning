# MAX Graph on Apple Silicon GPU - First Reported Success?

**Disclaimer:** I'm still learning MAX and ML infrastructure, so apologies if I've missed something obvious or if others have already done this!

## Key Finding

✅ **MAX Graph API with simple GPU kernels works on Apple Silicon (M1 Pro)!**

I managed to get element-wise operations running successfully on GPU via MAX Graph. Not sure if I'm the first, but haven't seen other reports in Discord, so thought I'd share in case it's useful for the team or others experimenting.

## What Works

- Element-wise operations: `ops.mul`, `ops.add`, `ops.relu`
- Graph compilation and execution on `Device(type=gpu,id=0)`
- Results validated correctly against NumPy

## Performance Observations

**Note:** I'm not confident my benchmarking approach is fair or optimal, so take these numbers with a grain of salt! I'm measuring simple element-wise ops across different tensor sizes:

- **4 elements:** CPU 19x faster (0.03ms vs 0.6ms)
- **1M elements:** CPU 9x faster (0.33ms vs 3.0ms)  
- **4M elements:** CPU 4.5x faster (0.95ms vs 4.3ms)
- **8M elements:** CPU 10x faster (2.6ms vs 27ms)

CPU dominates across all sizes tested. This likely reflects GPU dispatch overhead and how well small tensors fit in CPU cache. Not sure if there are better workloads or patterns to test where GPU would shine.

**Benchmark code:** https://github.com/DataBooth/max-learning/blob/main/benchmarks/01_elementwise/cpu_vs_gpu_scaling.py

Feedback on benchmarking methodology welcome!

## Current Blockers

- **No `matmul` kernel** for Apple Silicon GPU (error: "mma not supported")
- Transformers/DistilBERT won't work without matmul support
- GPU only viable for much larger workloads or more complex operations

## Xcode 26 Compatibility Issue

If you're on Xcode 26, you'll hit this error:
```
xcrun: error: unable to find utility "metallib", not a developer tool or in PATH
```

**Fix:** Explicitly download Metal Toolchain (~750MB):
```bash
xcodebuild -downloadComponent MetalToolchain
```

This is because Xcode 26 switched to on-demand component downloads.

## Code Example

Simple element-wise example (works on both CPU and GPU):
https://github.com/DataBooth/max-learning/blob/main/examples/python/01_elementwise/elementwise.py

## Environment

- **Hardware:** MacBook Pro 2021 (Apple M1 Pro, 16GB RAM)
- **Model:** MacBookPro18,3
- **OS:** macOS 15.7.3 (Darwin 24.6.0)
- **MAX:** 25.1.0

## Questions

1. Are there any other operations beyond element-wise that have Apple Silicon GPU kernels?
2. Any timeline or plans for matmul support on Apple GPUs?
3. Happy to test other operations or patterns if helpful for kernel development!

Thanks to the Modular team for the pointer that "some models in MAX might work" on Apple Silicon GPU - that was the encouragement to try this! 🙏
