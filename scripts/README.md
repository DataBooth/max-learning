# Scripts

## MAX Version Update (`update_max_version.py`)

Safely updates MAX to the latest nightly build and validates compatibility.

### Usage

```bash
python scripts/update_max_version.py
```

### What It Does

1. **Backs up** current `pixi.toml`
2. **Updates** to latest MAX nightly build
3. **Runs** full test suite (`pytest tests/ -v`)
4. **If tests pass**: Locks to new version in `pixi.toml`
5. **If tests fail**: Automatically rolls back to previous version and saves failure report

### Example Output

#### Successful Update

```
=== MAX Version Update & Test ===

📋 Backing up current pixi.toml...
   Current version: 26.1.0.dev2026010718

📝 Updating pixi.toml to latest nightly...
⏳ Update modular package...
✓ Update modular package succeeded
   New version: 26.2.0.dev2026012505

⏳ Run test suite with new version...
✓ Run test suite with new version succeeded

✓ All tests passed!

🔒 Locking to new version: 26.2.0.dev2026012505...

✓ Updated and locked to 26.2.0.dev2026012505

📊 Summary:
   Old version: 26.1.0.dev2026010718
   New version: 26.2.0.dev2026012505

Next steps:
  1. Review changes: git diff pixi.toml
  2. Update RELEASE_NOTES.md if needed
  3. Commit: git add pixi.toml && git commit -m 'chore: update MAX to 26.2.0.dev2026012505'
```

#### Failed Update (Automatic Rollback)

```
=== MAX Version Update & Test ===

📋 Backing up current pixi.toml...
   Current version: 26.1.0.dev2026010718

📝 Updating pixi.toml to latest nightly...
⏳ Update modular package...
✓ Update modular package succeeded
   New version: 26.2.0.dev2026012505

⏳ Run test suite with new version...
✗ Run test suite with new version failed

✗ Tests failed with new version!

📄 Test failure report saved to: test_failure_26.2.0.dev2026012505.txt

Test failures/errors summary:
  ERROR tests/python/01_elementwise/test_elementwise.py
  ImportError: cannot import name 'Tensor' from 'max.driver'
  ERROR tests/python/02_linear_layer/test_linear_layer.py
  ...

⏮️ Reverting to 26.1.0.dev2026010718...
⏳ Restore previous environment...
✓ Restore previous environment succeeded

⚠️  Rolled back to 26.1.0.dev2026010718

The new version (26.2.0.dev2026012505) broke compatibility.
Review test_failure_26.2.0.dev2026012505.txt for details.
```

### Why Use This Script?

MAX is under active development with frequent breaking API changes. This script:

- ✅ **Automates** the update and test workflow
- ✅ **Protects** against breaking changes by validating all tests
- ✅ **Documents** failures with detailed reports
- ✅ **Rolls back** automatically if compatibility breaks
- ✅ **Saves time** by avoiding manual testing and rollback steps

### Current Status (January 2026)

As of this writing, `26.2.0.dev2026012505` breaks compatibility due to `Tensor` being removed from `max.driver`. The script correctly detects this and maintains the working version `26.1.0.dev2026010718`.

### Test Failure Reports

When a version update fails tests, a detailed report is saved to `test_failure_<version>.txt` containing:

- Old and new version numbers
- Full pytest output with error messages
- Timestamp of the attempted update

These reports are gitignored but useful for:
- Understanding what broke
- Reporting issues to Modular
- Deciding whether to adapt code or wait for fixes
