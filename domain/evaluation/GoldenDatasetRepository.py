import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GoldenDatasetRepository:
    """Loads the hand-curated evaluation questions from a JSON file."""

    def __init__(self):
        self.path = os.path.join(BASE_DIR, "data", "golden_dataset.json")

    def load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Golden dataset not found at {self.path}")
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["samples"]