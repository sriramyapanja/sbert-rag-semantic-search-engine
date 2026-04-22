import os
from dotenv import load_dotenv
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


class RagEvaluator:
    """Wraps Ragas to score the RAG pipeline's outputs."""

    def __init__(self):
        # OpenAI as the judge LLM (cheap and reliable)
        self.evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-4o-mini", temperature=0)
        )
        # Reuse the same embedding model as the RAG pipeline
        self.evaluator_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )

    def _build_dataset(self, rag_outputs):
        samples = []
        for item in rag_outputs:
            samples.append(
                SingleTurnSample(
                    user_input=item["question"],
                    retrieved_contexts=item["contexts"],
                    response=item["answer"],
                    reference=item.get("reference"),
                )
            )
        return EvaluationDataset(samples=samples)

    def run(self, rag_outputs, use_reference_metrics=True):
        dataset = self._build_dataset(rag_outputs)

        metrics = [Faithfulness(), ResponseRelevancy()]
        if use_reference_metrics:
            metrics += [LLMContextPrecisionWithReference(), LLMContextRecall()]

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embeddings,
        )
        return result