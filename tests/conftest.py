import os
import tempfile

# Keep tests away from the real data/ directory.
os.environ["PILENS_DATA"] = tempfile.mkdtemp(prefix="pilens_test_")
