import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.update({
    "DATA_DIR": _tmp, "DETECTOR": "none", "EMBEDDER": "dummy", "API_KEY": "test-key",
    "MATCH_THRESHOLD": "0.8", "REVIEW_THRESHOLD": "0.5", "MIN_MARGIN": "0.05",
})
