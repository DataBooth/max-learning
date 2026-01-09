# Xcode Compatibility Issue

## Problem

MAX GPU compilation fails on macOS with Xcode 26.2 due to missing `metallib` tool.

## Environment

- **macOS**: Sonoma 14.6 (or later)
- **Xcode**: 26.2 (Build 17C52)
- **Hardware**: Apple Silicon M1
- **MAX**: Via pixi/modular package (January 2026)

## Error

```
max/kernels/src/Mogg/MOGGKernelAPI:1:1: error: Please make sure Xcode is installed and setup correctly
xcrun: error: unable to find utility "metallib", not a developer tool or in PATH
```

## Investigation

### What We Found

1. **Xcode is properly installed**:
   ```bash
   $ xcode-select -p
   /Applications/Xcode.app/Contents/Developer
   
   $ xcodebuild -version
   Xcode 26.2
   Build version 17C52
   ```

2. **Metal compiler exists**:
   ```bash
   $ xcrun -find metal
   /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal
   ```

3. **But `metallib` tool is missing**:
   ```bash
   $ which metallib
   metallib not found
   
   $ xcrun -find metallib
   xcrun: error: unable to find utility "metallib"
   ```

4. **Available Metal tools in Xcode 26.2**:
   ```
   metal
   metal-package-builder
   ```

### Root Cause

**Xcode 26.2 removed or renamed the `metallib` tool** that MAX expects for compiling Metal shaders.

This is a compatibility issue between:
- MAX's Metal toolchain expectations (designed for older Xcode)
- Xcode 26.2's updated Metal toolchain

## Workarounds

### Option 1: Wait for MAX Update

**Best option**: Wait for Modular to update MAX to support Xcode 26.2.

**Action**: Report this issue to Modular:
- [Modular GitHub Issues](https://github.com/modular/modular/issues)
- [Modular Discord](https://discord.gg/modular)

**Issue template**:
```
Title: MAX GPU compilation fails with Xcode 26.2 - metallib tool not found

Environment:
- Xcode 26.2 (Build 17C52)
- macOS Sonoma 14.6
- Apple Silicon M1
- MAX version: [check with `mojo --version`]

Error:
xcrun: error: unable to find utility "metallib", not a developer tool or in PATH

Description:
Element-wise GPU operations fail to compile because MAX is looking for the
`metallib` tool which no longer exists in Xcode 26.2. The tool appears to 
have been removed or renamed in this Xcode version.

Available Metal tools in Xcode 26.2:
- metal
- metal-package-builder

Expected: MAX should support the current Xcode toolchain or gracefully
fall back if unsupported tools are detected.
```

### Option 2: Downgrade Xcode (Not Recommended)

If you need GPU support urgently, you could try downgrading to Xcode 15 or 16:

```bash
# Download older Xcode from https://developer.apple.com/download/all/
# Install side-by-side
# Switch xcode-select path

sudo xcode-select --switch /Applications/Xcode_15.app/Contents/Developer
```

**Caution**: This affects your entire system's build environment.

### Option 3: CPU-Only (Current Approach)

Continue using CPU for now:
- ✅ Works perfectly
- ✅ 5.58x speedup over PyTorch already
- ✅ No toolchain issues

## Status

**As of January 2026**:
- ❌ Apple Silicon GPU not usable with MAX due to Xcode 26.2 incompatibility
- ✅ CPU inference works great
- ⏳ Waiting for MAX to support newer Xcode versions

## Timeline

- **January 7, 2026**: Xcode 26.2 released
- **January 9, 2026**: Issue discovered during GPU experiments
- **Future**: Awaiting MAX update

## Related Issues

- Element-wise operations have GPU kernels but can't compile
- Matrix operations (`matmul`) don't have GPU kernels yet (separate issue)

See: `examples/python/README_gpu_experiments.md` for GPU kernel availability status.

## References

- [Xcode Releases](https://xcodereleases.com/)
- [MAX Documentation](https://docs.modular.com/max/)
- [Metal Shader Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
