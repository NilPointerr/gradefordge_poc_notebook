# 🎓 GradeForge — AI Exam Paper Review Pipeline

An intelligent, grounded exam paper review and assessment system designed for student evaluations (grades 5–12 and beyond). GradeForge ingests textbook study material, answer keys, and student answer sheets, compares student answers against curriculum ground truth, and generates structured scoring and feedback reports with beautifully rendered mathematical formulas.

---

## 🌟 Key Features

* **Multi-Format Extraction via MinerU:** Structured Markdown and JSON representation preserving chapter hierarchies, headings, tables, and mathematical formulas.
* **Document-Aware Hierarchical Chunking:** 
  * Heading-aware semantic chunking for textbooks.
  * Question-aware parent-child chunking for answer keys and student papers (supporting bold and numbered questions).
* **Multi-Namespace Pinecone Vector DB:** Logical separation between `book_chunks`, `answer_key_chunks`, and `student_answer_chunks` to prevent retrieval cross-contamination.
* **Relevance Thresholding & Guardrails:** Hard cosine similarity thresholding (`SIMILARITY_THRESHOLD = 0.30`) and grounded system prompts to prevent out-of-domain hallucinations and gracefully handle out-of-curriculum questions.
* **Multi-LLM Auto-Fallback Architecture:**
  * **Primary:** Google Gemini (`gemini-3-flash-preview` / `gemini-2.5-flash`)
  * **Cloud Fallback:** Groq (`openai/gpt-oss-120b` / `llama-3.1-8b-instant`)
  * **Local Fallback:** Ollama (`qwen3:4b`)
* **Mathematical Formula Rendering Engine:**
  * Automatically detects display (`$$...$$`, `\[...\]`) and inline (`$...$`, `\(...\)`) LaTeX expressions.
  * Normalizes math notation and generates crisp 220-DPI transparent PNG formula images cached in `formula_cache/`.
  * Integrates TrueType `DejaVuSans` font in ReportLab to eliminate missing-glyph black boxes (`■`) for degrees (`°`), roots (`√`), and Greek symbols.
* **Automated PDF Report Generation:**
  * **Single-Question Assessment PDF:** Detailed scores, correctness, missing points, and textbook citations with embedded formula images.
  * **Comprehensive Full Exam Scorecard:** Multi-question scorecard summary, performance bands, and question-by-question breakdown.

---

## 🏗️ Architecture Pipeline

```text
       Textbook PDF / Answer Key PDF / Student Paper PDF
                             │
                             ▼
                    MinerU Extraction
                    (Markdown + JSON)
                             │
                             ▼
                Structure Parser & Classifier
             (book / answer_key / student_paper)
                             │
                             ▼
              Document-Aware Chunking & Metadata
                             │
                             ▼
          Sentence-Transformers (all-MiniLM-L6-v2)
                             │
                             ▼
         Pinecone Serverless Vector DB (Namespaces)
                             │
                             ▼
           Threshold Retrieval + Cross-Encoder Reranker
                             │
                             ▼
        Answer Review Agent (Gemini / Groq / Ollama)
                             │
                             ▼
         Formula Rendering Engine (Mathtext -> PNG)
                             │
                             ▼
       ReportLab PDF Reports (Score + Detailed Feedback)
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10+ (Python 3.12 recommended)
* A [Pinecone](https://www.pinecone.io/) API Key
* A [Google AI Studio (Gemini)](https://aistudio.google.com/) API Key
* (Optional) [Groq](https://groq.com/) API Key for cloud fallback
* (Optional) [Ollama](https://ollama.com/) running locally for local offline fallback

### 2. Installation

Clone the repository:
```bash
git clone https://github.com/NilPointerr/gradefordge_poc_notebook.git
cd gradefordge_poc_notebook
```

Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy the sample environment file:
```bash
cp .sample-env .env
```

Edit `.env` and fill in your API keys:
```env
PINECONE_API_KEY="your-pinecone-api-key"
GEMINI_API_KEY="your-gemini-api-key"
GEMINI_LLM_MODEL="gemini-3-flash-preview"

# Optional Fallbacks
GROQ_API_KEY="your-groq-api-key"
GROQ_LLM_MODEL="openai/gpt-oss-120b"
OLLAMA_MODEL="qwen3:4b"

RERANK_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
```

---

## 📖 Running the Pipeline

Open the main Jupyter notebook:
```bash
jupyter notebook src/student_test_review_pipeline_poc/gradeforge_rag_pipeline.ipynb
```

Follow the sequential cells:
1. **Setup & Configuration:** Loads environment variables and verifies API connectivity.
2. **Chunking & Parsing:** Ingests MinerU Markdown files and builds structured chunk hierarchies.
3. **Embeddings & Upsert:** Embeds chunks using `all-MiniLM-L6-v2` and indexes them into Pinecone namespaces.
4. **Answer Evaluation Demo:** Evaluates student questions with reference context and guardrails.
5. **PDF Generation Demo:** Compiles evaluations into formatted assessment reports with rendered formulas.

---

## 📂 Repository Structure

```text
├── .env                           # Local environment secrets (ignored by git)
├── .sample-env                     # Sample environment variable template
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project packaging configuration
├── README.md                       # Project documentation
├── GradeForge_RAG_Chunking_Architecture_and_Current_Status.md # Comprehensive architecture doc
└── src/
    ├── student_test_review_pipeline_poc/
    │   ├── gradeforge_rag_pipeline.ipynb   # Main end-to-end RAG notebook
    │   ├── mineru_output/                  # Extracted Markdown & JSON curriculum datasets
    │   └── review_test/                    # Generated student evaluation PDF reports
    └── test_data/                          # Test papers & sample formulas
```

---

## 📄 License

This project is licensed under the MIT License.