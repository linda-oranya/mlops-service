import logging, json
from collections import deque, Counter
from pythonjsonlogger import jsonlogger
from pathlib import Path



def setup_json_logger():
    logger = logging.getLogger("ml-service")
    logger.setLevel(logging.INFO)

    Path("artifacts").mkdir(parents=True, exist_ok=True)

    sh = logging.StreamHandler()
    sh.setFormatter(jsonlogger.JsonFormatter())

    fh = logging.FileHandler("artifacts/requests.jsonl")
    fh.setFormatter(jsonlogger.JsonFormatter())

    if not logger.handlers:
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger



class RollingStats:
    def __init__(self, maxlen=500):
        self.buffer = deque(maxlen=maxlen)

    def update(self, X):
        self.buffer.extend(X)

    def summary(self):
        if not self.buffer:
            return {"count": 0}
        n = len(self.buffer)
        # simple column-wise mean/std for the first few columns
        cols = len(self.buffer[0])
        means = []
        for j in range(cols):
            col = [row[j] for row in self.buffer]
            m = sum(col)/len(col)
            means.append(m)
        return {"count": n, "mean": means[:5]}
    def last_class_counts(self, preds):
        return Counter(preds)