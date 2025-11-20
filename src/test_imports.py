"""
Quick smoke test to verify all imports work.
Run this before downloading/processing to catch environment issues early.

Usage:
  python src/test_imports.py
"""

import sys

def test_imports():
    """Test all required imports."""
    errors = []
    
    # Core scientific stack
    try:
        import torch
        print(f"✓ torch {torch.__version__}")
    except ImportError as e:
        errors.append(f"✗ torch: {e}")
    
    try:
        import torchvision
        print(f"✓ torchvision {torchvision.__version__}")
    except ImportError as e:
        errors.append(f"✗ torchvision: {e}")
    
    try:
        import numpy as np
        print(f"✓ numpy {np.__version__}")
    except ImportError as e:
        errors.append(f"✗ numpy: {e}")
    
    try:
        import scipy
        print(f"✓ scipy {scipy.__version__}")
    except ImportError as e:
        errors.append(f"✗ scipy: {e}")
    
    try:
        from sklearn import __version__ as sklearn_version
        print(f"✓ scikit-learn {sklearn_version}")
    except ImportError as e:
        errors.append(f"✗ scikit-learn: {e}")
    
    try:
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__}")
    except ImportError as e:
        errors.append(f"✗ matplotlib: {e}")
    
    # Optional but nice-to-have
    try:
        import swa_gaussian
        print(f"✓ swa_gaussian (from GitHub)")
    except ImportError:
        print("⚠ swa_gaussian not found (optional; install via: pip install git+https://github.com/wjmaddox/swa_gaussian.git)")
    
    # Report
    print("\n" + "="*50)
    if errors:
        print("MISSING DEPENDENCIES:")
        for err in errors:
            print(f"  {err}")
        print("\nInstall missing packages via:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("✓ All required imports available!")
        print("You can proceed with downloading and conformal analysis.")
        return True

if __name__ == '__main__':
    success = test_imports()
    sys.exit(0 if success else 1)
