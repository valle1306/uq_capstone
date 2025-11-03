#!/bin/bash
# Test script to verify all imports work correctly on Amarel

echo "=========================================="
echo "Testing Classification Pipeline Imports"
echo "=========================================="

cd /scratch/$USER/uq_capstone

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate uq_capstone

# Set PYTHONPATH
export PYTHONPATH=/scratch/$USER/uq_capstone:$PYTHONPATH

echo ""
echo "PYTHONPATH: $PYTHONPATH"
echo "Current directory: $(pwd)"
echo ""

# Test imports
echo "Testing imports..."
python -c "
import sys
print('Python path:')
for p in sys.path:
    print(f'  {p}')
print()

print('Testing imports...')
try:
    from src.data_utils_classification import get_classification_loaders
    print('✓ data_utils_classification imported successfully')
except Exception as e:
    print(f'✗ data_utils_classification failed: {e}')
    exit(1)

try:
    from src.conformal_risk_control import ConformalRiskControl
    print('✓ conformal_risk_control imported successfully')
except Exception as e:
    print(f'✗ conformal_risk_control failed: {e}')
    exit(1)

print()
print('✓✓✓ All imports successful! ✓✓✓')
"

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
