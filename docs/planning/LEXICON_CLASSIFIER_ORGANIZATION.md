# Plan: Organize Lexicon Classifier

## Current State

The lexicon classifier (`src/mojo/lexicon_classifier/`) is from v0.1.0 and doesn't fit the current repository structure:

**What it is:**
- Pure Mojo sentiment classifier (no MAX Graph)
- Uses simple lexicon-based approach (word counting)
- v0.1.0 baseline before MAX Graph work
- Not part of the MAX Graph learning progression

**Where it is:**
- `src/mojo/lexicon_classifier/` - under src/, implying it's a library
- Referenced in milestones as "v0.1.0 - Lexicon-based Baseline"
- Has pixi tasks: `mojo-build`, `mojo-run`

## Problem

1. **Confusing location**: It's in `src/` but it's not really a reusable library for MAX Graph examples
2. **Not part of learning path**: Doesn't use MAX Engine/Graph, so doesn't fit the numbered examples
3. **Historical artifact**: Was a baseline before moving to MAX Graph, but now feels out of place
4. **No tests**: Unlike all Python examples, no tests for this Mojo code

## Options

### Option 1: Move to examples/mojo/00_baseline/
**Pros:**
- Makes it clear it's an example, not a library
- Fits the "example" paradigm
- Can be documented as "baseline before MAX Graph"

**Cons:**
- Not really part of the learning progression
- Numbered "00" feels forced

### Option 2: Move to examples/mojo/lexicon_baseline/
**Pros:**
- Clear that it's a baseline/reference implementation
- Still accessible as an example
- Name is descriptive

**Cons:**
- Still in examples but not teaching MAX Graph

### Option 3: Create separate comparison/ or baselines/ directory
**Pros:**
- Clear separation from MAX Graph examples
- Can include other baseline implementations
- Structure: `baselines/mojo_lexicon/`

**Cons:**
- Adds another top-level directory

### Option 4: Keep but document clearly
**Pros:**
- No refactoring needed
- Already works

**Cons:**
- Remains confusing
- Doesn't fit repository focus

### Option 5: Archive/deprecate
**Pros:**
- Simplifies repository focus on MAX Graph
- Less maintenance
- Can be in git history if needed

**Cons:**
- Loses working Mojo example
- Loses v0.1.0 baseline reference

## Recommendation

**Option 2: Move to `examples/mojo/lexicon_baseline/`**

**Reasoning:**
1. Keeps it accessible as a working example
2. Clear that it's not part of MAX Graph learning path
3. Provides baseline for comparison with MAX Graph implementations
4. Can be documented as "pure Mojo without MAX Graph"
5. Makes space for MAX Graph Mojo examples (01, 02, etc.)

## Implementation Plan

1. ✅ Create this plan document
2. ☐ Create `examples/mojo/lexicon_baseline/` directory
3. ☐ Move contents from `src/mojo/lexicon_classifier/` to `examples/mojo/lexicon_baseline/`
4. ☐ Create README.md explaining:
   - This is v0.1.0 baseline (no MAX Graph)
   - Pure Mojo sentiment classifier
   - Uses lexicon-based approach
   - Serves as baseline for comparing with MAX Graph examples
5. ☐ Update pixi.toml tasks:
   - Change paths from `src/mojo/lexicon_classifier/` to `examples/mojo/lexicon_baseline/`
6. ☐ Update main README.md:
   - Update repository structure diagram
   - Clarify v0.1.0 milestone description
   - Add note about baseline in examples
7. ☐ Remove `src/mojo/` directory (if empty)
8. ☐ Create examples/mojo/README.md:
   - Explain Mojo examples
   - Distinguish baseline vs MAX Graph examples
   - Point to future MAX Graph Mojo examples (01, 02)
9. ☐ Update MOJO_EXAMPLES_PLAN.md to reference baseline location

## Directory Structure After Move

```
examples/mojo/
├── lexicon_baseline/           # v0.1.0 baseline (no MAX Graph)
│   ├── classifier.mojo
│   ├── cli.mojo
│   ├── config.mojo
│   ├── config.toml
│   ├── main.mojo
│   ├── utils.mojo
│   └── README.md              # NEW: Explains this is non-MAX baseline
├── 01_elementwise/            # FUTURE: MAX Graph in Mojo
├── 02_linear_layer/           # FUTURE: MAX Graph in Mojo
└── README.md                  # NEW: Explains Mojo examples structure
```

## Benefits

1. **Clearer structure**: Examples are in examples/
2. **Better organization**: Baseline vs MAX Graph distinction clear
3. **Room for growth**: Makes space for MAX Graph Mojo examples
4. **Preserved history**: Keeps working v0.1.0 baseline accessible
5. **Documentation**: Forces clear explanation of what this code is/isn't

## Timeline

Can be done in v0.3.1 or v0.4.0 as housekeeping.

---

**Status**: Planning  
**Priority**: Low-Medium (improves organization but not urgent)  
**Complexity**: Low (mostly file moves and documentation)
