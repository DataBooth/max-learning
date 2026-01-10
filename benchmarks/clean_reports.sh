#!/usr/bin/env bash
#
# Clean benchmark report files
#
# Usage:
#   bash benchmarks/clean_reports.sh [md|json|csv|all]
#   pixi run clean-reports-md
#   pixi run clean-reports-json
#   pixi run clean-reports-csv
#   pixi run clean-reports-all

set -e

FORMAT="${1:-all}"

case "$FORMAT" in
    md)
        echo "🧹 Cleaning Markdown reports..."
        find benchmarks/*/results -name "*.md" -type f -delete 2>/dev/null || true
        echo "✓ Removed all .md files"
        ;;
    json)
        echo "🧹 Cleaning JSON reports..."
        find benchmarks/*/results -name "*.json" -type f -delete 2>/dev/null || true
        echo "✓ Removed all .json files"
        ;;
    csv)
        echo "🧹 Cleaning CSV reports..."
        find benchmarks/*/results -name "*.csv" -type f -delete 2>/dev/null || true
        echo "✓ Removed all .csv files"
        ;;
    all)
        echo "🧹 Cleaning all benchmark reports..."
        find benchmarks/*/results -type f \( -name "*.md" -o -name "*.json" -o -name "*.csv" \) -delete 2>/dev/null || true
        echo "✓ Removed all benchmark report files"
        ;;
    *)
        echo "❌ Invalid format: $FORMAT"
        echo "Usage: $0 [md|json|csv|all]"
        exit 1
        ;;
esac

echo ""
echo "Remaining files:"
find benchmarks/*/results -type f 2>/dev/null | wc -l | xargs echo "  Files:"
