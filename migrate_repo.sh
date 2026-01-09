#!/bin/bash
# migrate_repo.sh - Repository reorganisation script

set -e  # Exit on error

echo "🚀 Starting repository reorganisation..."
echo ""

# 1. Create directory structure
echo "📁 Creating new directory structure..."
mkdir -p src/python/max_distilbert
mkdir -p src/mojo/lexicon_classifier
mkdir -p examples/python examples/mojo
mkdir -p tests/python/integration tests/mojo
mkdir -p benchmarks/results benchmarks/test_data
mkdir -p config

# 2. Move Python MAX implementation
echo "🐍 Moving Python MAX implementation..."
if [ -d "src/max_distilbert" ]; then
    mv src/max_distilbert/* src/python/max_distilbert/
    rmdir src/max_distilbert
fi

# 3. Move Mojo lexicon classifier
echo "🔥 Moving Mojo lexicon classifier..."
[ -f src/classifier.mojo ] && mv src/classifier.mojo src/mojo/lexicon_classifier/
[ -f src/cli.mojo ] && mv src/cli.mojo src/mojo/lexicon_classifier/
[ -f src/config.mojo ] && mv src/config.mojo src/mojo/lexicon_classifier/
[ -f src/main.mojo ] && mv src/main.mojo src/mojo/lexicon_classifier/
[ -f src/utils.mojo ] && mv src/utils.mojo src/mojo/lexicon_classifier/

# 4. Delete redundant files
echo "🗑️  Deleting redundant files..."
rm -f src/embeddings.mojo
rm -f src/max_classifier.mojo
rm -f test_embeddings.py
rm -f test_simple.py
rm -f DEBUG_PLAN.md
rm -f PROGRESS_2026-01-08.md
rm -f SESSION_SUMMARY.md
rm -f REPO_ORGANISATION.md

# 5. Organise examples
echo "📚 Organising examples..."
[ -f examples/minimal_max_example.py ] && mv examples/minimal_max_example.py examples/python/minimal_max_graph.py

# 6. Organise tests
echo "🧪 Organising tests..."
[ -f tests/test_config.mojo ] && mv tests/test_config.mojo tests/mojo/test_lexicon_classifier.mojo

# 7. Organise benchmarks - KEEP ALL RESULTS
echo "📊 Organising benchmarks..."
[ -f benchmark.py ] && mv benchmark.py benchmarks/distilbert_max_vs_hf.py

# Move test data
if [ -d test_data ]; then
    mv test_data/* benchmarks/test_data/
    rmdir test_data
fi

# Move ALL benchmark results (keep datetime filenames)
if [ -d benchmark_results ]; then
    mv benchmark_results/* benchmarks/results/
    rmdir benchmark_results
fi

# 8. Organise configs
echo "⚙️  Organising configs..."
[ -f config.toml ] && mv config.toml config/lexicon_classifier.toml
[ -f benchmark_config.toml ] && mv benchmark_config.toml config/benchmark.toml

# 9. Consolidate docs
echo "📖 Consolidating docs..."
[ -f BLOG_DRAFT.md ] && mv BLOG_DRAFT.md docs/
[ -f MAX_VALUE_PROPOSITION.md ] && mv MAX_VALUE_PROPOSITION.md docs/

echo ""
echo "✅ Repository reorganisation complete!"
echo ""
echo "📝 Files created/updated next:"
echo "  - examples/python/README.md (with MAX docs URLs)"
echo "  - examples/python/distilbert_sentiment.py"
echo "  - Update benchmarks/distilbert_max_vs_hf.py paths"
echo ""
