#!/bin/bash
# Final cleanup script to remove all old directories
# Run with: bash scripts/final_cleanup.sh

set -e

echo "============================================================"
echo "Final Cleanup - Removing Old Directories"
echo "============================================================"
echo ""

# Fix permissions first
echo "Fixing permissions..."
chmod -R u+w results_* logs_* chinese_glm_* 2>/dev/null || true

# Remove old result directories
echo "Removing old result directories..."
for dir in results_* chinese_glm_*; do
    if [ -d "$dir" ]; then
        echo "  Removing: $dir"
        rm -rf "$dir" 2>/dev/null || echo "    ⚠️  Could not remove $dir (may need manual removal)"
    fi
done

# Remove old log directories
echo "Removing old log directories..."
for dir in logs_*; do
    if [ -d "$dir" ]; then
        echo "  Removing: $dir"
        rm -rf "$dir" 2>/dev/null || echo "    ⚠️  Could not remove $dir (may need manual removal)"
    fi
done

echo ""
echo "============================================================"
echo "Cleanup Complete!"
echo "============================================================"
echo ""
echo "Remaining directories:"
ls -1d */ 2>/dev/null | grep -v "^venv$\|^wandb$" | sed 's|/$||' | sort
