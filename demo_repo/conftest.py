import sys
from pathlib import Path

# Ensure `import billing` works regardless of pytest's rootdir/import-mode
# behavior, both when run directly and when run from inside a Verifier
# sandbox copy of this directory (AGENTS.md Section 5.4/5.6).
sys.path.insert(0, str(Path(__file__).resolve().parent))
