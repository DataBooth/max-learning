#!/usr/bin/env python3
"""
Safely update MAX to latest nightly and test.
Rolls back if tests fail.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path


class Colors:
    GREEN = "\033[0;32m"
    RED = "\033[0;31m"
    YELLOW = "\033[1;33m"
    RESET = "\033[0m"


def print_status(symbol: str, message: str, color: str = Colors.RESET) -> None:
    """Print coloured status message."""
    print(f"{color}{symbol}{Colors.RESET} {message}")


def get_modular_version() -> str:
    """Get currently installed Modular version via mojo."""
    result = subprocess.run(
        ["pixi", "run", "mojo", "--version"],
        capture_output=True,
        text=True,
    )
    # Format: "Mojo 0.26.1.0.dev2026010718 (5fdfb9e5)\n"
    match = re.search(r"(\d+\.\d+\.\d+\.dev\d+)", result.stdout)
    return match.group(1) if match else "unknown"


def read_locked_version(pixi_toml: Path) -> str:
    """Extract locked version from pixi.toml."""
    content = pixi_toml.read_text()
    match = re.search(r'^modular = "==([^"]+)"', content, re.MULTILINE)
    return match.group(1) if match else "unknown"


def update_pixi_toml(pixi_toml: Path, version: str | None = None) -> None:
    """Update pixi.toml to specific version or wildcard."""
    content = pixi_toml.read_text()

    if version is None:
        # Use wildcard to get latest
        new_content = re.sub(
            r'^modular = "==.*".*$',
            'modular = "*"',
            content,
            flags=re.MULTILINE,
        )
    else:
        # Lock to specific version
        new_content = re.sub(
            r'^modular = ".*".*$',
            f'modular = "=={version}"  # Locked to prevent breaking changes',
            content,
            flags=re.MULTILINE,
        )

    pixi_toml.write_text(new_content)


def run_command(cmd: list[str], description: str) -> tuple[bool, subprocess.CompletedProcess]:
    """Run command and return success status and result."""
    print_status("⏳", f"{description}...", Colors.YELLOW)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print_status("✓", f"{description} succeeded", Colors.GREEN)
        return True, result
    else:
        print_status("✗", f"{description} failed", Colors.RED)
        if result.stderr:
            print(f"   Error: {result.stderr.strip()}")
        return False, result


def main() -> int:
    """Main update workflow."""
    print(f"\n{Colors.YELLOW}=== MAX Version Update & Test ==={Colors.RESET}\n")

    # Setup paths
    repo_root = Path(__file__).parent.parent
    pixi_toml = repo_root / "pixi.toml"
    backup_file = repo_root / "pixi.toml.backup"

    # Step 1: Backup current pixi.toml
    print_status("📋", "Backing up current pixi.toml...")
    shutil.copy(pixi_toml, backup_file)
    current_version = read_locked_version(pixi_toml)
    print(f"   Current version: {current_version}\n")

    # Step 2: Update to wildcard (gets latest)
    print_status("📝", "Updating pixi.toml to latest nightly...")
    update_pixi_toml(pixi_toml, version=None)

    # Step 3: Update environment (with cache refresh)
    success, _ = run_command(["pixi", "update", "modular"], "Update modular package")
    if not success:
        print_status("⏮️", "Reverting to backup...", Colors.YELLOW)
        shutil.move(str(backup_file), str(pixi_toml))
        return 1

    # Step 4: Check new version
    new_version = get_modular_version()
    print(f"   New version: {new_version}\n")

    if new_version == current_version:
        print_status("ℹ️", f"Already on latest nightly ({new_version})", Colors.YELLOW)
        print("   No changes needed.\n")
        backup_file.unlink()
        return 0

    # Step 5: Run tests
    success, test_result = run_command(
        ["pixi", "run", "pytest", "tests/", "-v", "--tb=short"],
        "Run test suite with new version",
    )
    if not success:
        print(f"\n{Colors.RED}✗ Tests failed with new version!{Colors.RESET}\n")

        # Save failure report
        failure_report = repo_root / f"test_failure_{new_version}.txt"
        failure_content = f"""MAX Version Update Test Failure Report
{'=' * 60}

Attempted Update: {current_version} → {new_version}
Date: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}

Test Output:
{'-' * 60}
{test_result.stdout}

{test_result.stderr if test_result.stderr else ''}
"""
        failure_report.write_text(failure_content)

        print(f"📄 Test failure report saved to: {failure_report.name}\n")

        # Extract failed test names from pytest output
        failed_tests = []
        for line in test_result.stdout.split("\n"):
            if "FAILED" in line:
                failed_tests.append(line.strip())

        # Show errors/failures summary
        error_lines = [
            line for line in test_result.stdout.split("\n") if "ERROR" in line or "FAILED" in line
        ]
        if error_lines:
            print(f"{Colors.RED}Test failures/errors summary:{Colors.RESET}")
            for line in error_lines[:15]:  # Show first 15
                if line.strip():
                    print(f"  {line}")
            if len(error_lines) > 15:
                print(f"  ... and {len(error_lines) - 15} more\n")
            else:
                print()

        print_status("⏮️", f"Reverting to {current_version}...", Colors.YELLOW)
        shutil.move(str(backup_file), str(pixi_toml))
        run_command(["pixi", "install"], "Restore previous environment")

        print(f"\n{Colors.YELLOW}⚠️  Rolled back to {current_version}{Colors.RESET}\n")
        print(f"The new version ({new_version}) broke compatibility.")
        print(f"Review {failure_report.name} for details.\n")
        return 1

    # Step 6: Lock to new version
    print(f"\n{Colors.GREEN}✓ All tests passed!{Colors.RESET}\n")
    print_status("🔒", f"Locking to new version: {new_version}...")
    update_pixi_toml(pixi_toml, version=new_version)

    print(f"\n{Colors.GREEN}✓ Updated and locked to {new_version}{Colors.RESET}\n")
    print("📊 Summary:")
    print(f"   Old version: {current_version}")
    print(f"   New version: {new_version}\n")
    print("Next steps:")
    print("  1. Review changes: git diff pixi.toml")
    print("  2. Update RELEASE_NOTES.md if needed")
    print(f"  3. Commit: git add pixi.toml && git commit -m 'chore: update MAX to {new_version}'\n")

    backup_file.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
