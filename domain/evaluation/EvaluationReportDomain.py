import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EvaluationReport:
    """Saves Ragas evaluation results to timestamped CSV files."""

    def __init__(self):
        self.output_dir = os.path.join(BASE_DIR, "data", "eval_results")
        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, result):
        df = result.to_pandas()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.output_dir, f"ragas_eval_{timestamp}.csv")
        df.to_csv(csv_path, index=False)

        # Compute averages per metric
        non_metric_cols = {"user_input", "retrieved_contexts", "response", "reference"}
        metric_cols = [c for c in df.columns if c not in non_metric_cols]
        summary = df[metric_cols].mean().to_dict()

        return csv_path, summary