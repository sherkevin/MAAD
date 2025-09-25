#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("Current working directory:", os.getcwd())
print("Python path:")
for p in sys.path:
    print("  ", p)

try:
    import numpy as np
    print("numpy version:", np.__version__)
    print("numpy imported successfully")
except ImportError as e:
    print("numpy import failed:", e)

try:
    import pandas as pd
    print("pandas version:", pd.__version__)
    print("pandas imported successfully")
except ImportError as e:
    print("pandas import failed:", e)

try:
    import sklearn
    print("sklearn version:", sklearn.__version__)
    print("sklearn imported successfully")
except ImportError as e:
    print("sklearn import failed:", e)

print("Environment test completed")
