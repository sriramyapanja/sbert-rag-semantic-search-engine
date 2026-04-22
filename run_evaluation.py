import requests
from domain.evaluation.GoldenDatasetRepository import GoldenDatasetRepository
from domain.evaluation.RagEvaluationDomain import RagEvaluator
from domain.evaluation.EvaluationReportDomain import EvaluationReport

RAG_API_URL = "http://127.0.0.1:5000/service/search"


def query_rag(question):
    resp = requests.post(RAG_API_URL, json={"sentence_query": question}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["answer"], data["contexts"]


def main():
    print("Loading golden dataset...")
    golden = GoldenDatasetRepository().load()
    print("Loaded " + str(len(golden)) + " test questions.")

    print("Running RAG pipeline on each question...")
    rag_outputs = []
    for i, item in enumerate(golden, 1):
       print("  [" + str(i) + "/" + str(len(golden)) + "] " + item["question"])
       answer, contexts = query_rag(item["question"])
       rag_outputs.append({
           "question": item["question"],
            "answer": answer,
             "contexts": contexts,
             "reference": item.get("reference"),
      })

    print("Evaluating with Ragas (this takes a few minutes)...")
    evaluator = RagEvaluator()
    result = evaluator.run(rag_outputs, use_reference_metrics=True)

    csv_path, summary = EvaluationReport().save(result)

    print("Done. Results saved to: " + csv_path)
    print("=== Average scores ===")
    for metric, score in summary.items():
     print("  " + metric.ljust(45) + " " + "{:.3f}".format(score))


if __name__ == "__main__":
  main()