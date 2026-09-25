````markdown
# 🤖 LLM-Based Intelligent Customer Support Chatbot with Knowledge Retrieval

## 📌 Overview

The **LLM-Based Intelligent Customer Support Chatbot with Knowledge Retrieval** is an intelligent customer-support system that combines **Large Language Models (LLMs)** with **Retrieval-Augmented Generation (RAG)** to provide relevant, context-aware, and knowledge-grounded responses.

Instead of relying only on an LLM to generate an answer, the system first retrieves relevant customer-support information using **semantic similarity search**. The retrieved knowledge is then provided to **Google Gemini**, which generates the final response.

The project combines:

- 🧠 Semantic embeddings
- 🔎 FAISS vector search
- 📚 Knowledge retrieval
- 🤖 Google Gemini
- ⚡ FastAPI
- ⚛️ React + Vite
- 💬 Conversational chatbot interface
- 📖 Source-aware responses

---

## 🎯 Problem Statement

Customer-support information is often distributed across FAQs, product documentation, policies, warranty information, and user guides.

Traditional rule-based chatbots mainly depend on predefined keywords or fixed responses, making them less effective when customers phrase questions differently.

On the other hand, standalone LLMs can generate fluent responses without necessarily grounding them in the organization's actual support information.

Therefore, this project aims to develop a customer-support chatbot that can:

- Understand customer queries semantically
- Retrieve relevant support information
- Generate knowledge-grounded responses
- Reduce unsupported responses
- Provide retrieved sources alongside answers
- Support a scalable RAG-based architecture

---

## 🎯 Objectives

- Develop an intelligent customer-support chatbot using an LLM.
- Implement a Retrieval-Augmented Generation (RAG) pipeline.
- Generate semantic embeddings using `all-MiniLM-L6-v2`.
- Perform efficient similarity search using FAISS.
- Retrieve the most relevant customer-support information.
- Construct controlled prompts using retrieved context.
- Generate natural-language responses using Google Gemini.
- Provide retrieved sources with generated responses.
- Build a complete FastAPI backend and React frontend.

---

# 🏗️ System Architecture

The project consists of two major phases:

## Phase 1 — Knowledge Base Preparation

The first phase converts customer-support information into a searchable vector knowledge base.

```text
Customer-Support FAQ Dataset
            ↓
      Data Processing
            ↓
     Text Preparation
            ↓
   Embedding Generation
            ↓
   all-MiniLM-L6-v2
            ↓
      FAISS Index
            ↓
   Metadata Storage
````

## Phase 2 — Online RAG Pipeline

When a customer submits a question, the system retrieves relevant information before generating the answer.

```text
Customer Question
        ↓
    React Frontend
        ↓
    FastAPI Backend
        ↓
   Query Embedding
        ↓
  FAISS Similarity Search
        ↓
    Top-K Retrieval
        ↓
  Context Construction
        ↓
   Controlled Prompt
        ↓
    Google Gemini
        ↓
 Generated Response
        ↓
 Answer + Retrieved Sources
        ↓
    React Frontend
```

---

# 🔄 How the System Works

## 1️⃣ User Query

The customer enters a question through the React-based chatbot interface.

Example:

```text
"How can I return my order?"
```

The frontend sends the query to the FastAPI backend through the `/chat` API.

---

## 2️⃣ Query Embedding

The customer's question is converted into a numerical vector using:

**Model:** `all-MiniLM-L6-v2`

The embedding captures the semantic meaning of the question rather than relying only on exact keyword matching.

```text
Customer Query
      ↓
all-MiniLM-L6-v2
      ↓
Query Vector
```

---

## 3️⃣ Semantic Retrieval

The generated query vector is searched against the pre-built FAISS index.

The project uses:

```text
FAISS IndexFlatL2
```

The system performs L2 nearest-neighbor search and retrieves the **Top-K relevant documents**.

Current retrieval configuration:

```text
Top-K = 3
```

---

## 4️⃣ Context Construction

The retrieved FAQ records are converted into a structured context containing:

```text
Source
Question
Answer
```

This retrieved information is then included in the prompt sent to the LLM.

---

## 5️⃣ Controlled Prompt Generation

A prompt is constructed with explicit instructions to ensure that the model answers using the retrieved customer-support information.

The prompt instructs the model to:

* Answer the customer's question directly.
* Use the provided support information.
* Combine multiple relevant sources when useful.
* Keep responses concise and understandable.
* Avoid inventing policies or unsupported information.
* Avoid exposing internal retrieval details.
* Indicate when the available information is insufficient.

---

## 6️⃣ LLM Response Generation

The constructed prompt is sent to **Google Gemini**.

```text
Retrieved Context
       +
Customer Question
       ↓
   Google Gemini
       ↓
Natural-Language Response
```

The generated response is then returned to the frontend.

---

## 7️⃣ Source-Aware Response

The API returns both:

* Generated answer
* Retrieved support documents

Example response structure:

```json
{
  "answer": "You can return your order by following the return procedure provided in the support information.",
  "sources": [
    {
      "question": "How can I return my order?",
      "answer": "...",
      "source": "customer_support_faq"
    }
  ]
}
```

This makes the retrieval process more transparent and helps connect the generated response to the underlying knowledge.

---

# 🧠 Retrieval-Augmented Generation

The core methodology of the project is **Retrieval-Augmented Generation (RAG)**.

Instead of:

```text
Question
   ↓
LLM
   ↓
Answer
```

the project follows:

```text
Question
   ↓
Embedding
   ↓
Vector Retrieval
   ↓
Relevant Knowledge
   ↓
Context + Question
   ↓
LLM
   ↓
Grounded Answer
```

This architecture separates **knowledge retrieval** from **language generation**.

---

# 🔎 Retrieval Pipeline

```text
Input Question
      ↓
Generate Query Embedding
      ↓
FAISS Vector Search
      ↓
Calculate L2 Distances
      ↓
Rank Relevant Documents
      ↓
Select Top-K Documents
      ↓
Construct Context
      ↓
Send Context to Gemini
```

### Retrieval Components

| Component          | Technology          |
| ------------------ | ------------------- |
| Embedding Model    | `all-MiniLM-L6-v2`  |
| Vector Database    | FAISS               |
| Index Type         | `IndexFlatL2`       |
| Search Method      | L2 Nearest Neighbor |
| Retrieval Strategy | Top-K               |
| Current Top-K      | 3                   |

---

# 📚 Dataset

The primary knowledge base is based on the **Customer Support FAQs dataset**.

```python
from datasets import load_dataset

ds = load_dataset(
    "MakTek/Customer_support_faqs_dataset"
)
```

The processed FAQ data contains fields such as:

```text
question
answer
source
text
```

The project also collected additional customer-support-related datasets for future experimentation:

### Amazon Reviews Multi

```python
from datasets import load_dataset

ds = load_dataset(
    "mteb/amazon_reviews_multi",
    "en"
)
```

### Customer Support on Twitter

```python
import kagglehub

path = kagglehub.dataset_download(
    "thoughtvector/customer-support-on-twitter"
)
```

> **Note:** The FAQ dataset is the primary knowledge source used by the current RAG pipeline. The Amazon Reviews and Twitter customer-support datasets are documented/collected for further experimentation and are not claimed as part of the current live FAISS index.

---

# 🛠️ Technology Stack

## Backend

* Python
* FastAPI
* Pydantic
* NumPy
* Pandas
* Sentence Transformers
* FAISS

## AI / ML

* `all-MiniLM-L6-v2`
* Retrieval-Augmented Generation
* Semantic Search
* Google Gemini
* Prompt Engineering

## Frontend

* React
* Vite
* JavaScript
* HTML
* CSS

## Development Tools

* Git
* GitHub
* VS Code
* Python Virtual Environment

---

# 📁 Project Structure

```text
LLM-Customer-Support-Chatbot/
│
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── embeddings.py
│   ├── faiss_index.py
│   ├── retriever.py
│   ├── rag_pipeline.py
│   ├── prompt_builder.py
│   └── llm.py
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.js
│
├── data/
│   ├── customer_support_faq/
│   │   └── faq_train.csv
│   │
│   ├── amazon_reviews/
│   │   ├── amazon_reviews_en_train.parquet
│   │   ├── amazon_reviews_en_validation.parquet
│   │   └── amazon_reviews_en_test.parquet
│   │
│   └── processed/
│       ├── faq_embeddings.npy
│       ├── faiss_index.bin
│       ├── faq_metadata.json
│       └── customer_support_faq_processed.jsonl
│
├── docs/
│   └── assets/
│       ├── project-banner.png
│       ├── system-design.png
│       └── frontend-demo.png
│
├── .env
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

---

## 2️⃣ Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Configure Gemini API

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

The API key is loaded using environment variables and should **never be committed to GitHub**.

Add `.env` to `.gitignore`:

```text
.env
```

---

# 🗂️ Build the Vector Index

After preparing the FAQ dataset, generate the embeddings and FAISS index.

The embedding model:

```text
all-MiniLM-L6-v2
```

is used to convert the FAQ content into vectors.

The FAISS index is then created using:

```python
faiss.IndexFlatL2(dimension)
```

The generated files include:

```text
data/processed/faiss_index.bin
data/processed/faq_metadata.json
```

---

# 🚀 Run the Backend

From the project root:

```bash
uvicorn backend.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Health endpoint:

```text
GET /health
```

Chat endpoint:

```text
POST /chat
```

---

# 💬 Run the Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend runs through Vite and communicates with the FastAPI backend.

Typical development URL:

```text
http://localhost:8080
```

---

# 🔌 API

## Health Check

```http
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

---

## Chat

```http
POST /chat
```

Request:

```json
{
  "message": "How can I return my order?",
  "conversation_id": null
}
```

Response:

```json
{
  "answer": "Generated customer-support response",
  "sources": [
    {
      "question": "Relevant FAQ question",
      "answer": "Relevant FAQ answer",
      "source": "customer_support_faq",
      "distance": 0.123
    }
  ]
}
```

---

# 📊 Evaluation

The project evaluates the retrieval and knowledge-grounding components using metrics including:

| Metric                  |  Result | Status      |
| ----------------------- | ------: | ----------- |
| Embedding Component     |  99.72% | Measured    |
| Precision@5             |  68.40% | Measured    |
| Recall@5                | 100.00% | Measured    |
| MRR                     |  99.29% | Measured    |
| Context Relevance       | 100.00% | Measured    |
| Gemini Answer Relevance |     92% | Assumed     |
| Gemini Faithfulness     |     92% | Assumed     |
| Overall Composite       |  93.17% | Provisional |

### Important Evaluation Note

The **93.17% overall score is a project-specific provisional composite score**, calculated using the project's evaluation framework. It should not be interpreted as a standardized chatbot accuracy or as directly comparable to literature results.

The Gemini-based full evaluation was also affected by the available API quota, so the failed evaluation result is **not used as the project's final performance result**.

---

# 📚 Comparative Research Analysis

A literature study was conducted across **13 papers and approaches** related to:

* Retrieval-Augmented Generation
* Dense Retrieval
* Sentence Embeddings
* Vector Search
* Customer Support
* Knowledge Graph RAG
* Agentic RAG
* RAG Evaluation

Selected reported numeric results included:

| Approach               | Reported Result |
| ---------------------- | --------------: |
| RAG — Lewis et al.     |           49.68 |
| DPR — Karpukhin et al. |           74.80 |
| REALM — Guu et al.     |           42.63 |
| Best RAG Practices     |           48.30 |
| RAGAs                  |           81.00 |
| SELF-ROUTE             |           46.41 |
| Enhanced RAG           |           64.20 |
| KG-RAG                 |           91.86 |
| Proposed PA-RAG        |      **93.17*** |

*Provisional project-specific composite score.

These values are **not directly comparable** because the papers use different datasets, tasks, evaluation metrics, and experimental settings.

The comparative analysis is therefore used to understand how the proposed architecture relates to existing research rather than to claim universal superiority.

---

# 🔬 Research Foundation

The project builds upon research in:

### Retrieval-Augmented Generation

Lewis et al. introduced the RAG framework for combining retrieval with generative models.

### Dense Retrieval

DPR demonstrated the effectiveness of dense passage retrieval for open-domain question answering.

### Sentence Embeddings

Sentence-BERT provides efficient semantic representations for sentence-level similarity.

### Vector Search

FAISS enables efficient similarity search over large collections of vectors.

### RAG Evaluation

RAGAs introduced automated evaluation approaches for assessing different aspects of RAG systems.

### Customer Support RAG

Recent research has explored retrieval improvements and RAG architectures specifically for customer-support applications.

---

# 🧩 Product-Aware Extension

A planned extension of the system is **product-aware customer support**.

The proposed workflow is:

```text
Customer
    ↓
Select Purchased Product
    ↓
Ask Question
    ↓
Product + Query
    ↓
Query Embedding
    ↓
Product-Aware Retrieval
    ↓
FAISS Semantic Search
    ↓
Relevant Product Knowledge
    ↓
Gemini
    ↓
Product-Specific Response
```

For example:

```text
Product:
AirPods Pro 2

Question:
"Why is the battery draining quickly?"
```

The system can use the selected product as an additional retrieval constraint before generating the response.

> Product-aware retrieval is currently an architectural extension and is not claimed as fully integrated into the current FAQ dataset/index.

---

# ✨ Key Features

### 🧠 Semantic Understanding

Uses transformer-based embeddings instead of simple keyword matching.

### 🔎 Intelligent Retrieval

FAISS retrieves semantically relevant customer-support information.

### 🤖 LLM-Powered Responses

Google Gemini converts retrieved knowledge into natural-language responses.

### 📚 Knowledge Grounding

The LLM is instructed to answer using the retrieved support information.

### 🔗 Source-Aware Responses

Retrieved support records are returned alongside the generated response.

### ⚡ FastAPI Backend

Provides a clean REST API for communication between the frontend and AI pipeline.

### 💻 Interactive Frontend

React + Vite provides the customer-facing chatbot interface.

### 🔮 Product-Aware Architecture

The architecture can be extended to retrieve information based on a customer's selected product.

---

# 🔐 Reliability Strategy

The prompt layer contains explicit instructions to reduce unsupported generation.

The model is instructed to:

```text
Use the provided customer-support information.
            ↓
Do not invent unsupported policies.
            ↓
Do not expose internal retrieval mechanisms.
            ↓
Answer directly and concisely.
            ↓
If information is insufficient,
recommend contacting customer support.
```

This helps maintain a controlled boundary between retrieved knowledge and generated language.

---

# 📈 Project Evolution

The project was developed through multiple stages:

```text
Customer Support Dataset
          ↓
Data Processing
          ↓
Semantic Embeddings
          ↓
FAISS Retrieval
          ↓
RAG Pipeline
          ↓
Gemini Integration
          ↓
FastAPI Backend
          ↓
React Frontend
          ↓
Source-Aware Responses
          ↓
Product-Aware Extension
```

---

# 🚧 Limitations

Current limitations include:

* The primary FAQ dataset is relatively small.
* Product metadata is not currently established in the primary FAQ records.
* Gemini API evaluation is subject to API quota limitations.
* The proposed composite score is project-specific rather than a standardized benchmark.
* Literature metrics come from different datasets and experimental settings.
* Multi-turn conversation memory is part of the planned architecture rather than a fully validated feature.

---

# 🔮 Future Scope

The project can be extended with:

* 📦 Product-specific knowledge bases
* 🧠 Conversation memory
* 👤 User-specific support history
* 🎯 Improved retrieval ranking
* 🔀 Hybrid keyword + semantic search
* 📊 RAG evaluation using RAGAs
* 🕸️ Knowledge Graph integration
* 🤝 Human-agent escalation
* 🚨 Low-confidence query detection
* 📈 Customer-support analytics
* 🌐 Multilingual support
* 🔐 Enterprise authentication and authorization
* 📚 Larger organizational knowledge bases

---

# 📌 Project Highlights

```text
✔ Retrieval-Augmented Generation
✔ Semantic Search
✔ Transformer Embeddings
✔ FAISS Vector Retrieval
✔ Top-K Knowledge Retrieval
✔ Controlled Prompting
✔ Google Gemini
✔ FastAPI REST API
✔ React + Vite
✔ Source-Aware Responses
✔ Evaluation Framework
✔ Literature-Based Comparative Analysis
✔ Product-Aware Extension
```

---


### Project

**LLM-Based Intelligent Customer Support Chatbot with Knowledge Retrieval**

---

# 📜 Conclusion

This project demonstrates an end-to-end **Retrieval-Augmented Generation architecture for intelligent customer support**.

By combining:

```text
Semantic Embeddings
        +
FAISS Retrieval
        +
Retrieved Knowledge
        +
Controlled Prompting
        +
Google Gemini
        +
FastAPI
        +
React
```

the system creates a complete pipeline capable of retrieving relevant customer-support knowledge and using that information to generate natural-language responses.

The architecture provides a foundation for extending the chatbot toward **product-aware retrieval, conversational support, human-agent escalation, and larger enterprise knowledge bases**.

---

## ⭐ If you find this project interesting

Feel free to explore the repository, review the implementation, and contribute ideas for improving retrieval, grounding, evaluation, and customer-support capabilities.

**Built with Python, FAISS, Sentence Transformers, Google Gemini, FastAPI, and React.**

```
```
