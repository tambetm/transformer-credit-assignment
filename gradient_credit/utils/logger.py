"""Simple CSV logger for training metrics."""

import os
import csv


class CSVLogger:
    """Log training metrics to CSV files."""

    def __init__(self, log_dir: str, filename: str = "metrics.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, filename)
        self.file = None
        self.writer = None
        self.fieldnames = None

    def log(self, metrics: dict):
        """Log a row of metrics. Initializes headers on first call."""
        if self.writer is None:
            self.fieldnames = list(metrics.keys())
            self.file = open(self.filepath, "w", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()

        self.writer.writerow(metrics)
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
