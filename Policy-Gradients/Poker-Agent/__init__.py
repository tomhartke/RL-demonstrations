import sys
from pathlib import Path

# Removes the need for .imports of e.g. env_wrapper
HERE = Path(__file__).parent.resolve()
sys.path.append(str(HERE))
