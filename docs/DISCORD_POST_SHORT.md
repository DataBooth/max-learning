## MAX Graph on Apple Silicon GPU - First reported success? 🎉

✅ **Got MAX Graph API working on Apple Silicon GPU (M1 Pro)** for element-wise operations (`ops.mul`, `ops.add`, `ops.relu`)!

**Key learnings:**
- Graph compiles and runs correctly on `Device(type=gpu,id=0)`
- Results validate against NumPy
- **Blocker:** No `matmul` kernel yet ("mma not supported") - transformers won't work until this lands

**Xcode 26 gotcha:** Need to explicitly download Metal Toolchain:
```bash
xcodebuild -downloadComponent MetalToolchain
```

**Code & benchmarks:** https://github.com/DataBooth/max-learning/blob/main/docs/APPLE_SILICON_GPU_FINDINGS.md

**Questions:**
1. Any other ops with Apple Silicon GPU kernels?
2. Timeline/plans for matmul support?
3. Happy to test if helpful!

*Disclaimer: Still learning MAX - apologies if I've missed something obvious!*

**Env:** MacBook Pro 2021 (M1 Pro, 16GB) | macOS 15.7.3 | MAX 25.1.0
