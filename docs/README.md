# 🤖 LLM-Based Intelligent Customer Support Chatbot with Knowledge Retrieval

An AI-powered customer support chatbot that combines **Retrieval-Augmented Generation (RAG), semantic search, FAISS vector retrieval, and Google Gemini** to retrieve relevant FAQ information and generate context-aware responses.

## 📌 Overview

Customer support systems often handle repetitive questions about products, services, policies, and common issues. Finding the right information manually can be time-consuming, while standalone Large Language Models (LLMs) may generate answers that are not grounded in the actual support knowledge base.

This project addresses these challenges by combining semantic retrieval with LLM-based response generation. It retrieves relevant FAQ records using vector similarity search and supplies the retrieved context to Google Gemini to generate a response.

The application uses **React + Vite** for the frontend and **FastAPI** for the backend.

## 🎯 Problem Statement

Traditional customer support systems may rely on keyword matching or predefined responses, making it difficult to handle questions expressed in different ways. Standalone LLMs may also generate unsupported information when they lack access to the relevant knowledge source.

The objective of this project is to develop a chatbot that retrieves semantically relevant customer-support information and uses the retrieved context to generate helpful, context-aware responses.

## ✨ Key Features

- 🧠 **Semantic Search:** Identifies relevant FAQ entries based on meaning rather than exact keyword matches.
- 🔎 **FAISS Vector Retrieval:** Searches the FAQ embedding index using `IndexFlatL2`.
- 📚 **Knowledge-Grounded Generation:** Supplies retrieved FAQ context to the language model.
- 🤖 **Google Gemini Integration:** Generates natural-language responses from the user query and retrieved context.
- ⚡ **FastAPI Backend:** Exposes API endpoints for communication between the frontend and AI pipeline.
- 💻 **React + Vite Frontend:** Provides a user-facing chatbot interface.
- 🔗 **Retrieved Sources:** The API response is designed to return retrieved support records alongside the generated answer.
- 🧩 **Extensible Architecture:** Provides a foundation for future improvements such as product-aware retrieval and human-agent escalation.

## 🏗️ System Architecture

The system follows a Retrieval-Augmented Generation workflow.

### 1. Knowledge Base Preparation

```text
Customer Support FAQ Dataset
           ↓
    Data Processing
           ↓
  Text Representation
           ↓
  Embedding Generation
           ↓
   all-MiniLM-L6-v2
           ↓
      FAISS Index
           ↓
    FAQ Metadata
```

### 2. Online Query Processing

```text
      User Question
            ↓
       React + Vite
            ↓
       FastAPI API
            ↓
      Query Embedding
            ↓
     FAISS Similarity Search
            ↓
     Top-3 FAQ Retrieval
            ↓
      Context Construction
            ↓
      Controlled Prompt
            ↓
       Google Gemini
            ↓
     Generated Response
            ↓
       Chatbot Interface
```

## 🔄 How It Works

### 1. User Query

The user enters a customer-support question through the chatbot interface.

Example:

```text
How can I return my order?
```

The frontend sends the question to the FastAPI backend.

### 2. Query Embedding

The question is converted into a numerical vector using the `all-MiniLM-L6-v2` sentence-transformer model.

This representation helps the system identify semantically similar FAQ records even when the query wording differs from the stored question.

### 3. Semantic Retrieval

FAISS searches the FAQ vector index using `IndexFlatL2`, which performs exact nearest-neighbour search based on squared L2 distance.

The current runtime retrieval configuration selects the top three records.

### 4. Context Construction

The retrieved FAQ records provide relevant information for response generation. The context may contain fields such as:

- Question
- Answer
- Source

### 5. Prompt Construction

The user query and retrieved information are assembled into a prompt for Google Gemini.

The prompt is designed to encourage the model to use the provided support information, avoid inventing unsupported policies, and indicate when the available context is insufficient.

### 6. Response Generation

Google Gemini generates a natural-language response based on the query and retrieved context.

### 7. Response Delivery

The backend returns the generated answer to the frontend. The API response can also include retrieved FAQ records to make the information used by the retrieval pipeline more transparent.

## 🧠 Retrieval-Augmented Generation (RAG)

Instead of sending a question directly to an LLM, the system first retrieves relevant information.

```text
User Question
      ↓
Query Embedding
      ↓
Vector Similarity Search
      ↓
Relevant FAQ Records
      ↓
Question + Retrieved Context
      ↓
Google Gemini
      ↓
Generated Response
```

This separates knowledge retrieval from language generation and provides the LLM with relevant external information when generating a response.

## 📚 Dataset

The primary knowledge source is the **Customer Support FAQs dataset**.

Dataset identifier:

`MakTek/Customer_support_faqs_dataset`

Example loading code:

```python
from datasets import load_dataset

dataset = load_dataset(
    "MakTek/Customer_support_faqs_dataset"
)
```

The main FAQ knowledge base used by the current retrieval pipeline contains **200 records**.

The processed FAQ records include fields such as question, answer, and source, depending on the available dataset schema.

### Additional Datasets

The following datasets were also collected for potential future experimentation:

**Amazon Reviews Multi**

```python
from datasets import load_dataset

dataset = load_dataset(
    "mteb/amazon_reviews_multi",
    "en"
)
```

**Customer Support on Twitter**

```python
import kagglehub

path = kagglehub.dataset_download(
    "thoughtvector/customer-support-on-twitter"
)
```

**Dataset status:** The FAQ knowledge base is the primary source used by the current RAG pipeline. The Amazon Reviews Multi and Customer Support on Twitter datasets are not claimed to be integrated into the live FAISS index.

Dataset sources and licences should be reviewed before redistribution.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | React, Vite, JavaScript, HTML, CSS |
| Backend | FastAPI, Uvicorn |
| Embedding Model | `all-MiniLM-L6-v2` |
| Vector Search | FAISS |
| Index Type | `IndexFlatL2` |
| LLM | Google Gemini API |
| NLP Approach | Semantic Search, Embeddings, RAG |
| Data Processing | NumPy, Pandas |
| Development Tools | Git, GitHub, Visual Studio Code |

## 📁 Project Structure

The following is the intended logical structure. Confirm the filenames against the current repository before relying on this tree.

```text
LLM-Customer-Support-Chatbot/
│
├── backend/
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
│   └── processed/
│
├── docs/
│   └── assets/
│
├── .env
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### Prerequisites

Install the following before running the application:

- Python 3.x
- Node.js and npm
- Git
- A Google Gemini API key
- Access to the project source code and required dataset files

### 1. Clone the Repository

```bash
git clone https://github.com/Pravallika2789/-KLH-CSIT-2026-2027-2420090073-CustomerSupportChatBot.git

cd -KLH-CSIT-2026-2027-2420090073-CustomerSupportChatBot
```

If the repository directory name differs after cloning, enter the directory created by Git.

### 2. Create a Python Virtual Environment

**Windows:**

```bash
python -m venv .venv

.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### 3. Install Backend Dependencies

From the project root, run:

```bash
pip install -r requirements.txt
```

If the project uses separate dependency files, follow the instructions provided in the corresponding directory.

### 4. Configure the Gemini API Key

Create a `.env` file in the project root or in the location expected by the backend.

```env
GEMINI_API_KEY=your_api_key_here
```

Replace the placeholder with your own API key.

**Security:** Never commit API keys, credentials, or private configuration files to GitHub. Add `.env` to `.gitignore`.

### 5. Prepare the FAISS Index

Ensure that the FAQ data and any required processed files are available.

The embedding model used is:

```text
all-MiniLM-L6-v2
```

The vector index uses:

```python
faiss.IndexFlatL2(dimension)
```

If the repository does not contain the prebuilt index, run the project's index-generation script, if available, to generate the required index and metadata files.

### 6. Run the Backend

From the project root, activate the virtual environment and run:

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

This command assumes the backend package is named `backend` and contains `main.py`. Adjust the command if your repository uses a different entry point.

Backend address:

```text
http://127.0.0.1:8000
```

If FastAPI's interactive documentation is enabled, open:

```text
http://127.0.0.1:8000/docs
```

### 7. Run the Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite in the terminal. Ensure that the frontend API configuration points to the running FastAPI backend.

## 🔌 API Overview

The following endpoints describe the intended API interface. Confirm the actual routes and response schemas in the current backend implementation.

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy"
}
```

### Chat Endpoint

```http
POST /chat
```

Example request:

```json
{
  "message": "How can I return my order?",
  "conversation_id": null
}
```

Example response structure:

```json
{
  "answer": "Generated customer-support response",
  "sources": [
    {
      "question": "Relevant FAQ question",
      "answer": "Relevant FAQ answer",
      "source": "customer_support_faq"
    }
  ]
}
```

These JSON examples illustrate the interface structure; actual field names and values depend on the implemented API.

## 📊 Retrieval Evaluation

The retrieval component was evaluated using the following recorded results:

| Metric | Result | Status |
|---|---:|---|
| Precision@5 | 68.40% | Measured |
| Recall@5 | 100.00% | Measured |
| Mean Reciprocal Rank (MRR) | 99.29% | Measured |
| Context Relevance | 100.00% | Project-defined metric |

### Interpretation

- **Precision@5:** Measures the proportion of the top-five retrieved records that are relevant.
- **Recall@5:** Measures the proportion of relevant records retrieved within the top five.
- **MRR:** Measures how highly the first relevant result is ranked.
- **Context Relevance:** A project-defined measure whose calculation method should be documented for reproducibility.

These results describe the retrieval component and depend on the evaluation queries and relevance judgements.

### Generation Evaluation Status

The complete Gemini-based generation evaluation was not completed because API requests encountered free-tier quota errors (HTTP 429).

Therefore, the following results are **not available as validated measurements**:

- Generated-answer faithfulness
- Generated-answer relevance
- End-to-end answer accuracy
- User satisfaction

An earlier project-specific composite score and assumed generation scores are not presented as validated performance results.

## 🔬 Research Foundation

The project draws on established research in:

- **Retrieval-Augmented Generation:** Combining document retrieval with language generation.
- **Dense Retrieval:** Representing queries and documents as vectors for semantic search.
- **Sentence Embeddings:** Generating vector representations for sentence-level similarity.
- **Vector Search:** Searching dense vectors efficiently using FAISS.
- **RAG Evaluation:** Assessing retrieval quality and the relationship between retrieved context and generated responses.

Selected references:

1. Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*  
   https://arxiv.org/abs/2005.11401

2. Reimers and Gurevych (2019), *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.*  
   https://arxiv.org/abs/1908.10084

3. Johnson, Douze, and Jégou (2017), *Billion-scale similarity search with GPUs.*  
   https://arxiv.org/abs/1702.08734

4. Vaswani et al. (2017), *Attention Is All You Need.*  
   https://arxiv.org/abs/1706.03762

5. Devlin et al. (2019), *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*  
   https://arxiv.org/abs/1810.04805

## 🧩 Future Scope

Potential improvements include:

- 📦 Product-specific knowledge retrieval
- 🎯 Improved retrieval ranking and precision
- 🔀 Hybrid keyword and semantic search
- 🧠 Conversation memory
- 📊 Complete generation evaluation
- 🕸️ Knowledge graph integration
- 🤝 Human-agent escalation
- 🚨 Low-confidence query detection
- 📈 Customer-support analytics
- 🌐 Multilingual support
- 🔐 Authentication and authorization
- 📚 Support for larger organizational knowledge bases

Product-aware retrieval is a proposed extension and has not been validated as part of the current implementation.

## 🚧 Limitations

- The primary FAQ knowledge base contains 200 records.
- The exact dataset provenance and licence should be documented.
- Additional collected datasets are not confirmed as part of the live FAISS index.
- Gemini API quota limitations prevented completion of generation evaluation.
- Retrieval metrics do not establish the correctness of generated responses.
- Product-aware retrieval remains a proposed extension.
- Performance on larger datasets and concurrent requests has not been fully evaluated.

## 📌 Project Highlights

- Retrieval-Augmented Generation (RAG)
- Semantic text embeddings
- FAISS vector similarity search
- Top-K FAQ retrieval
- Controlled prompt construction
- Google Gemini API integration
- FastAPI backend
- React + Vite frontend
- Retrieved-source support
- Retrieval evaluation
- Extensible customer-support architecture

## 📜 Conclusion

This project demonstrates an integrated Retrieval-Augmented Generation architecture for customer support. By combining semantic embeddings, FAISS vector retrieval, retrieved FAQ context, controlled prompting, Google Gemini, FastAPI, and React, the system provides a foundation for generating context-aware customer-support responses.

The recorded retrieval evaluation provides an initial assessment of the search component. Further work is needed to improve retrieval precision and complete the evaluation of generated responses before making claims about end-to-end reliability.

## 👩‍💻 Author

**Reddy Gayatri Satya Sai Pravallika**

B.Tech – Computer Science and Information Technology  
K L (Deemed to be) University

## 🔗 Project Repository

[LLM-Based Intelligent Customer Support Chatbot with Knowledge Retrieval](https://github.com/Pravallika2789/-KLH-CSIT-2026-2027-2420090073-CustomerSupportChatBot)

Built with Python, Sentence Transformers, FAISS, Google Gemini, FastAPI, and React.
