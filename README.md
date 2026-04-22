# Semantic Search API with S-BERT and RAG/LLM

> A Python search engine that finds California government purchase records 
> by meaning, not keywords and measures how well it does it.

---

## Table of Contents

- [Highlights](#highlights)
- [Credit](#credit)
- [Screenshots](#screenshots)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Module 3 - Ragas Evaluation (Personal Extension)](#module-3--ragas-evaluation-personal-extension)
- [API Reference](#api-reference)
- [Takeaways](#takeaways)
- [License](#license)

---

## Highlights

- **Semantic search over 344K records** — Find records by meaning, not 
  exact words. "Fuel" surfaces "gasoline" and "diesel" automatically.
- **Two search modes** A ranked-list S-BERT search, and a natural 
  language Q&A mode powered by RAG + GPT-4o-mini.
- **Tolerant to messy input** Handles synonyms, plurals, casing, 
  stopwords, lemmatization, and light typos out of the box.
- **Measured, not guessed**  Module 3 adds a Ragas evaluation layer 
  with a golden dataset and four standard RAG metrics, so changes to 
  the pipeline can be tracked with real numbers.
- **Clean architecture**  Domain-Driven Design layout keeps business 
  logic, APIs, and infrastructure cleanly separated.
- **Self-documenting APIs**  Flasgger auto-generates Swagger UI so 
  every endpoint is testable in the browser.

---

## Credit

This project is my work through the Udemy course **"Using Artificial 
Intelligence (NLP) to build a semantic text query API with BERT and RAG 
(LangChain/LLM)"** by **André Vieira de Lima**  a Systems Analyst and 
Data Scientist at SERPRO (Brazil's Federal Data Processing Service). 
Modules 1 and 2 follow his architecture and teaching closely.

**Module 3 (Ragas evaluation) is a personal extension** I built after 
finishing the course, using tools and patterns not covered in the material.

![Udemy Certificate](screenshots/udemy-certificate.jpg)

---

## Screenshots

### Module 1 - S-BERT Semantic Search

| Swagger Home | Try It Out | Result |
|---|---|---|
| ![Swagger Home](screenshots/01-sbert-swagger-home.png) | ![Try it out](screenshots/02-sbert-swagger-try-it-out.png) | ![Result](screenshots/03-sbert-swagger-result-200.png) |

### Module 2 - RAG Contextual Search

| Swagger Home | Try It Out | Result |
|---|---|---|
| ![RAG Swagger Home](screenshots/04-rag-swagger-home.png) | ![Try it out](screenshots/05-rag-swagger-try-it-out.png) | ![Result](screenshots/06-rag-swagger-result-200.png) |

### Module 3 - Ragas Evaluation Output

![Ragas Scores](08-ragas-evaluation-scores.png)

---

## Technology Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.10 |
| **Data** | Pandas, NumPy, Apache Parquet |
| **NLP preprocessing** | NLTK (stopwords), SpaCy (`en_core_web_lg` for lemmatization) |
| **Embeddings** | Sentence Transformers — `all-mpnet-base-v2` |
| **Vector search** | FAISS |
| **LLM orchestration** | LangChain |
| **LLM provider** | OpenAI (GPT-4o-mini) |
| **API layer** | Flask + Flasgger (auto Swagger docs) |
| **Evaluation** *(my extension)* | Ragas |

---

## Installation

### Prerequisites

- **Python 3.10** (other versions are not tested)
- **HuggingFace account** (free), for downloading models
- **OpenAI API key**  for Module 2 
  generation and Module 3 judging

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/semantic-search-sbert-rag-llm.git
cd semantic-search-sbert-rag-llm

# 2. Create and activate a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download SpaCy model
python -m spacy download en_core_web_lg

# 5. Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"
```

### Environment Variables

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
OPENAI_API_KEY=your_openai_key_here
```

>  `.env` is included in `.gitignore` and should never be committed.

### Dataset

Download **Purchase Order Data 2012–2015** from the [California Open Data 
Portal](https://data.ca.gov/dataset/purchase-order-data) and place the 
CSV file in the `data/` folder. The dataset (~344K rows) is not bundled 
with this repository due to its size.

---

## Usage

### Step 1 - Preprocess the Data (one-time)

```bash
python domain/purchases/PurchaseOrderPreprocessingDomain.py
```

This cleans the raw CSV (removes nulls, anomalies, stopwords, lemmatizes, 
lowercases) and saves a compressed Parquet file. On the full dataset this 
takes 30–60 minutes due to SpaCy lemmatization.

### Step 2 - Generate Embeddings (one-time)

```bash
python domain/purchases/PurchaseOrderWordEmbeddingsDomain.py
```

Produces a serialized `.pkl` file of 768-dim sentence vectors.

### Step 3 - Run Module 1 (S-BERT Search)

```bash
python application/SemanticSearchApi.py
```

Open Swagger UI at [http://127.0.0.1:5000/apidocs](http://127.0.0.1:5000/apidocs) and try:

```json
{"sentence_query": "transport"}
```

**Expected response:**

```json
[
  { "score": 100.0, "item_name": "Transport",     "total_price": "$1,250.00" },
  { "score":  86.0, "item_name": "Transportation", "total_price": "$890.50"  },
  { "score":  74.0, "item_name": "Vehicle hire",   "total_price": "$412.00"  }
]
```

### Step 4 — Run Module 2 (RAG Q&A)

```bash
python application/SearchRagApi.py
```

In Swagger, send a natural language question:

```json
{"sentence_query": "Are there any computer services available?"}
```

**Expected response:**

```json
{
  "answer": "Yes, IT infrastructure maintenance and support services are available, including software development and maintenance.",
  "contexts": [
    "IT infrastructure maintenance and support",
    "Software development and maintenance services"
  ]
}
```

### Step 5 - Run Module 3 (Ragas Evaluation)

With the Module 2 API still running in one terminal, open a second terminal:

```bash
python run_evaluation.py
```

The script runs 15 golden-dataset questions through the live API, scores 
each response with Ragas, prints averages, and saves a timestamped CSV 
to `data/eval_results/`. Full run takes 3–5 minutes.

---

## Project Structure

This project follows **Domain-Driven Design** — business logic is 
organized by domain, not by file type.

```
semantic-search-sbert-rag-llm/
├── application/                     # Flask API layer
│   ├── Home.py
│   ├── SemanticSearchApi.py         # Module 1 endpoints
│   └── SearchRagApi.py              # Module 2 endpoint
├── domain/
│   ├── purchases/                   # Course modules
│   │   ├── PurchaseOrderRepository.py
│   │   ├── PurchaseOrderStatistics.py
│   │   ├── PurchaseOrderPreprocessingDomain.py
│   │   ├── PurchaseOrderWordEmbeddingsDomain.py
│   │   └── PurchaseOrderDomain.py
│   └── evaluation/                  # My extension (Module 3)
│       ├── GoldenDatasetRepository.py
│       ├── RagEvaluationDomain.py
│       └── EvaluationReportDomain.py
├── data/
│   ├── contracted_services.csv      # RAG corpus
│   ├── golden_dataset.json          # 15 eval Q&A pairs
│   └── eval_results/                # Ragas score CSVs
├── screenshots/
├── run_evaluation.py                # Module 3 entry point
├── requirements.txt
├── .env                             # Not committed
└── .gitignore
```

---

## Module 3 - Ragas Evaluation (Personal Extension)

After finishing the course I realized the project had no way to answer 
the question: *"is the RAG output actually good?"* Any change to the 
pipeline — different embedding model, chunk size, LLM, prompt — had no 
measurable signal behind it. Module 3 closes that gap.

### Approach

- **Golden dataset** - 15 hand-written question/reference pairs grounded 
  in the services corpus (`data/golden_dataset.json`).
- **Judge LLM** - GPT-4o-mini, on a separate call path from the 
  generation LLM.
- **Metrics** - the four standard Ragas RAG metrics.

### Results (15-sample run)

| Metric | Score | What It Measures |
|---|---|---|
| Faithfulness | **0.82** | Does the answer stick to the retrieved context? |
| Answer Relevancy | **0.77** | Does the answer address the question directly? |
| Context Precision | **0.93** | Are relevant chunks ranked above irrelevant ones? |
| Context Recall | **0.83** | Did retrieval pull in everything needed to answer? |

Retrieval is the strongest link — FAISS on a small, clean corpus 
behaves predictably well. Answer relevancy is the weakest; the course 
prompts the LLM to answer in three sentences, so answers sometimes 
include extra context beyond the literal question.

### Caveats

- 15 samples is small. Scores should be read as directional, not precise.
- The corpus itself is only 15 services, so context precision is easier 
  to score high on than it would be on a noisy production corpus.
- GPT-4o-mini as judge occasionally returned fewer generations than 
  requested, adding some noise to individual rows.

The point was not the numbers — it was having a reproducible loop where 
any future pipeline change can be measured, not guessed.

---

## API Reference

### Module 1 - S-BERT

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check / welcome |
| `GET` | `/purchases` | Return first 10 records |
| `GET` | `/purchase/<id>` | Lookup by order number |
| `POST` | `/purchaseorder/preprocessing` | Trigger preprocessing |
| `POST` | `/purchaseorder/semanticsearch` | Semantic search by meaning |

### Module 2 — RAG

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/service/search` | Natural language Q&A over services corpus |

### Module 3 — Evaluation

Not an HTTP endpoint. Runs as a batch script:

```bash
python run_evaluation.py
```

---

## Takeaways

**What the course taught me.** The end-to-end shape of an NLP pipeline 
— cleaning messy real-world text, encoding words as vectors with 
transformers, fast similarity search with FAISS, grounding LLM output 
in retrieved data via RAG, organizing code with DDD, and exposing ML 
behind a documented REST API.

**What adding Ragas taught me.** Building a model is only half the job. 
Without evaluation, you cannot tell whether a change is an improvement 
or a regression. LLM-as-a-judge is a reasonable way to get signal 
without hand-labeling thousands of examples, but the choice of judge 
model and the size of the eval set both matter, and scores should 
always be read as directional.

---

## License

This project is released under the **MIT License** — see the [LICENSE](LICENSE) 
file for details.

The course material, including the overall architecture of Modules 1 
and 2, is the intellectual property of the instructor André Vieira de 
Lima. This repository is my own implementation work done while following 
the course, shared for learning and portfolio purposes.

---

<p align="center">
  Built while transitioning from desktop support into AI-powered full-stack development.
</p>
