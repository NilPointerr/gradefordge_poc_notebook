# GradeForge — Exam Paper Review Agent
## End-to-End Architecture & Current Status

**Project:** GradeForge / Student Test Review Pipeline  
**Purpose:** Build a production-oriented exam answer review agent for students (targeting roughly grades 5–12) that can extract questions and answers from PDFs, compare student answers against correct/reference answers and study material, and generate scores and useful feedback.

---

# 1. Current Objective

The system should accept:

1. **Book / study-material PDF**
2. **Correct Q&A / answer-key PDF**
3. **Student question-answer / answer-sheet PDF**

and produce, for each question:

- identified question
- student answer
- expected/reference answer
- relevant supporting information from the book
- score
- explanation/reasoning
- missing points / mistakes
- suggestions for improvement

The system must handle content such as:

- normal text
- multiple languages
- tables
- diagrams/images
- mathematical equations/formulas
- potentially handwritten student answers

The main priority is **accurate extraction and grounded evaluation**, not simply generating an answer using an LLM.

---

# 2. Current Pipeline

The high-level pipeline is currently planned as:

```text
PDF
  |
  v
MinerU
  |
  +------------------+
  |                  |
  v                  v
Markdown             JSON
  |                  |
  +---------+--------+
            |
            v
      Structure Parser
            |
            v
     Document Classifier
            |
      +-----+------+----------------+
      |            |                |
      v            v                v
    Book       Answer Key      Student Paper
      |            |                |
      v            v                v
 Heading-aware   Q&A-aware       Q&A-aware
 Chunking        Chunking        Chunking
      |            |                |
      +------------+----------------+
                   |
                   v
             Metadata Layer
                   |
                   v
              Embeddings
                   |
                   v
              Vector DB
                   |
                   v
              Retrieval
                   |
                   v
           Answer Review Agent
                   |
                   v
          Score + Feedback
```

---

# 3. Current Status

## Completed / Solved

### 3.1 PDF Extraction

**MinerU has been selected and is currently being used for extraction.**

The previous extraction problem was addressed by moving to MinerU.

MinerU provides:

- Markdown output
- structured JSON output
- document structure information
- page/content relationships
- extraction of text and document elements

Therefore, the project does **not** need to solve the basic PDF-to-text extraction problem again at the RAG stage.

### 3.2 Important Architecture Decision

The **original PDF should not be treated as the primary chunking representation**.

Instead:

```text
Original PDF
     |
     v
   MinerU
     |
     +---- Markdown
     |
     +---- JSON
```

Markdown/JSON become the canonical structured representation for parsing and chunking.

The original PDF remains important for:

- source/provenance
- page references
- visual verification
- image/diagram references
- extraction validation
- final evidence/citation

### 3.3 RAG Pipeline Implementation & Latest Features (Updated October 2026)

**The full RAG pipeline has been implemented** in the notebook:

`src/student_test_review_pipeline_poc/gradeforge_rag_pipeline.ipynb`

This notebook implements the complete end-to-end pipeline:

- ✅ **Markdown Structure Parser** — Parses MinerU markdown into heading-aware blocks with hierarchy tracking
- ✅ **Document Classifier** — Multi-signal heuristic classifier (filename, Q&A density, heading structure, answer markers) → `book`, `answer_key`, or `student_paper`
- ✅ **Canonical Chunk Schema** — `Chunk` dataclass with all required metadata fields (chunk_id, parent_id, document_id, document_type, chunk_type, question_id, chapter, section, subsection, source, etc.)
- ✅ **Book Chunker** — Heading-aware + hierarchy-preserving + semantic chunking (300–800 token target)
- ✅ **Q&A Chunker** — Question-aware hierarchical chunking with parent-child relationships (used for both answer keys and student papers)
- ✅ **Chunk Type Detection** — Automatically identifies: concept, definition, formula, example, procedure, figure, table
- ✅ **Question Detection (v2)** — Robust multi-signal detection handling numbered patterns, heading patterns, keyword markers, and markdown formatting prefixes (`**1. ...**`, `*1. ...*`)
- ✅ **Embedding Generation** — `all-MiniLM-L6-v2` via sentence-transformers (384-dim)
- ✅ **Pinecone Vector DB** — Serverless index with namespace separation (`book_chunks`, `answer_key_chunks`, `student_answer_chunks`)
- ✅ **Retrieval & Relevance Thresholding** — Multi-namespace semantic search with configurable cosine similarity thresholding (`SIMILARITY_THRESHOLD = 0.30`) to reject irrelevant chunks
- ✅ **Reranking** — Cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for precision relevance scoring
- ✅ **Multi-LLM Review Agent with Auto-Fallback** — Unified Answer Review Agent:
  - **Primary:** Google Gemini (`gemini-3-flash-preview` / `gemini-2.5-flash`)
  - **Fallback 1:** Groq (`openai/gpt-oss-120b` / `llama-3.1-8b-instant`)
  - **Fallback 2:** Local Ollama (`qwen3:4b`)
- ✅ **Anti-Hallucination Guardrails** — Grounded evaluation prompt enforcing strict curriculum boundaries; out-of-domain questions are detected and flagged without fabricating criteria
- ✅ **Mathematical Formula Rendering Engine** — Automatically detects LaTeX formulas (`$$...$$`, `\[...\]`, `$...$`, `\(...\)`), normalizes mathtext notation, renders high-DPI transparent PNGs (`formula_cache/`), and embeds them cleanly into PDF reports
- ✅ **Unicode & Special Character Support** — TrueType `DejaVuSans` font integration preventing missing-glyph black squares (`■`) for degrees (`°`), roots (`√`), and Greek symbols
- ✅ **PDF Report Generation** — Two comprehensive ReportLab generators:
  - Single-question student assessment PDF (`generate_review_pdf`)
  - Full exam scorecard PDF with question-by-question breakdown (`generate_full_exam_pdf`)
- ✅ **Ingestion Utility** — `ingest_new_document()` function to process and upsert any new markdown file

**Tech stack used:**

- `sentence-transformers` for embeddings
- `pinecone` for vector DB
- `google-genai` (Gemini API) as primary evaluation LLM
- `groq` for cloud fallback LLM
- `ollama` for local fallback LLM
- `matplotlib` for LaTeX mathtext formula rendering
- `reportlab` with `DejaVuSans` TrueType font for PDF generation
- `pypdf` / `pypdfium2` for PDF inspection and page rendering
- `python-dotenv` for config management

---

# 4. Current RAG Design Decision

Before implementing the RAG pipeline, the first major decision is the **chunking strategy**.

A single generic chunker should NOT be used for every document.

There are three major document types:

```text
1. Book / Study Material
2. Correct Answer / Answer Key
3. Student Answer Sheet
```

Each requires a different primary chunking strategy.

---

# 5. Document Type A — Book / Study Material

## Primary Strategy

Use:

**Heading-aware + hierarchy-aware + semantic chunking**

Example:

```text
Chapter
  |
  +-- Section
       |
       +-- Subsection
            |
            +-- Paragraphs
            +-- Formula
            +-- Example
            +-- Table
            +-- Figure
```

Example Markdown:

```markdown
# Chapter 5: Motion

## 5.1 Introduction

Motion is the change in position...

## 5.2 Types of Motion

### Linear Motion

Linear motion is...

### Circular Motion

Circular motion is...

## 5.3 Equations of Motion

The equations are:

v = u + at
```

The heading hierarchy should be preserved in metadata.

A chunk should retain information such as:

```json
{
  "chapter": "Chapter 5: Motion",
  "section": "5.2 Types of Motion",
  "subsection": "Linear Motion",
  "content": "Linear motion is..."
}
```

## Important

Heading alone is NOT enough.

If one section contains thousands of tokens, it must be further split.

Therefore:

```text
Heading hierarchy
       |
       v
Content blocks
       |
       v
Semantic / size-based split
       |
       v
Final retrieval chunks
```

A target range around **300–800 tokens** can be used as a starting point, but this should be treated as a guideline rather than a rigid rule.

Structure should take priority over arbitrary token size.

---

# 6. Document Type B — Correct Answer / Answer Key

## Primary Strategy

Use:

**Question-aware hierarchical chunking**

Example:

```text
Q1
 |
 +-- Correct Answer

Q2
 |
 +-- Correct Answer

Q3
 |
 +-- Correct Answer
```

The question should be the primary parent boundary.

Example:

```json
{
  "question_id": "Q2",
  "question": "Explain the process of photosynthesis.",
  "answer": "..."
}
```

If the answer is small, the whole Q&A can remain one retrieval unit.

If the answer is very large:

```text
Q2
 |
 +-- Answer Chunk 1
 +-- Answer Chunk 2
 +-- Answer Chunk 3
```

Every child chunk must retain:

```text
question_id = Q2
parent_id = Q2
```

This prevents the relationship between a question and its reference answer from being lost.

---

# 7. Document Type C — Student Answer Sheet

## Primary Strategy

Also use:

**Question-aware hierarchical chunking**

Example:

```text
Q1
 |
 +-- Student Answer

Q2
 |
 +-- Student Answer

Q3
 |
 +-- Student Answer
```

The entire student answer should remain associated with its question.

If a student's answer is large, it may be split into child chunks, but all chunks retain the same:

```text
question_id
parent_id
```

This is important because grading requires the complete context of the student's response.

---

# 8. Question Detection Strategy

Do NOT rely on one keyword.

A question may appear as:

```text
Q1.
Q.1
Question 1
Que. 1
1.
1)
(1)
1]
```

Likewise, answers may use:

```text
Answer:
Ans:
Solution:
Sol:
```

or may have no explicit answer marker.

Therefore, Q&A detection should use multiple signals.

## Proposed signals

```text
1. Numbering pattern
2. "Q", "Question", "Que" markers
3. Heading level
4. Question-like language
5. Answer markers
6. Document position
7. Surrounding block structure
8. MinerU layout/element information
```

Conceptual pipeline:

```text
MinerU JSON/MD
      |
      v
Block Parser
      |
      v
Question Candidate Detection
      |
      +-- numbering
      +-- heading
      +-- keyword
      +-- language/question pattern
      +-- layout
      |
      v
Question Boundary
      |
      v
Answer Boundary
      |
      v
Q&A Object
```

The parser should be designed to support multiple exam-paper formats.

---

# 9. Parent-Child Chunking

A major design decision is to use hierarchical chunks rather than only flat chunks.

Example for a question:

```text
Q12
 |
 +-- Question
 |
 +-- Answer
       |
       +-- Chunk 1
       +-- Chunk 2
       +-- Chunk 3
```

Example for a book:

```text
Chapter 5
 |
 +-- Section 5.2
       |
       +-- Concept Chunk
       +-- Formula Chunk
       +-- Example Chunk
       +-- Figure Chunk
```

This allows retrieval at different levels without losing document relationships.

---

# 10. Chunk Types

The system should explicitly identify chunk types.

Possible types:

```text
concept
definition
formula
example
procedure
figure
table
question
answer
qa
student_answer
```

Not every document will contain every type.

### Book

Typical:

```text
concept
definition
formula
example
figure
table
```

### Answer Key

Typical:

```text
qa
answer
formula
figure
table
```

### Student Paper

Typical:

```text
qa
student_answer
formula
figure
table
```

---

# 11. Formula Handling

Formulas should not be accidentally separated from their explanation.

Example:

```text
The velocity is given by:

v = u + at

where:

u = initial velocity
v = final velocity
a = acceleration
t = time
```

The formula and its immediate explanation should remain related.

A formula can have its own typed element:

```json
{
  "type": "formula",
  "content": "v = u + at",
  "context": "Equation for velocity",
  "page": 42
}
```

The formula can also be linked to its parent concept/section.

The exact formula extraction technology is considered part of the extraction layer; MinerU is currently responsible for the initial extraction representation.

## 11.1 Formula Rendering Engine (Implemented — October 2026)

Passing raw LaTeX code (e.g. `\text{Attention}(Q,K,V)=\text{softmax}\!\Big(\frac{QK^{\top}}{\sqrt{d_k}}\Big)V` or `\(\sqrt{d_k}\)`) into student PDFs makes the assessment unreadable. GradeForge provides an automated **Formula Rendering Engine**:

```text
Markdown Review / Chunk Formula
               │
               ▼
   Regex Formula Tokenizer
   - Display: $$...$$ or \[...\]
   - Inline: $...$ or \(...\)
               │
               ▼
    LaTeX Normalizer
    - Cleans \operatorname, \Big, \!, \begin{array}, \tag, etc.
               │
               ▼
 Matplotlib Mathtext Renderer (220 DPI)
               │
               ▼
   PNG Cache (formula_cache/eq_xxx.png / inl_xxx.png)
               │
               ▼
 ReportLab Flowable / Inline <img> Embedding
```

1. **Dual Formula Types:**
   - **Display Formulas:** Centered, prominent formula image flowables with optional unicode subtitle captions.
   - **Inline Formulas:** Sized inline images (`<img src="..." height="11" valign="-2"/>`) embedded directly into sentence flow.
2. **TrueType Unicode Font Integration:**
   - ReportLab's standard Helvetica font cannot render math glyphs (`∘`, `√`, `θ`, `Δ`), causing black boxes (`■`).
   - GradeForge registers and uses `DejaVuSans` TrueType font across all PDF styles and normalizes OCR artifacts (such as converting ring operators `\u2218` to degree `°` and cleaning zero-width spaces).
3. **Textbook Image Crop Integration:**
   - MinerU equation crops (from `images/`) are mapped to chunk metadata (`image_path`), allowing authentic textbook crops to be attached alongside rendered equations.

---

# 12. Tables and Figures

Tables and figures should not be treated as ordinary paragraphs.

For example:

```json
{
  "type": "figure",
  "caption": "Structure of a cell",
  "page": 38,
  "description": "...",
  "image_reference": "..."
}
```

A table should similarly preserve its structure rather than flattening everything into arbitrary text chunks.

The final chunk representation should retain a reference to the original page/element.

---

# 13. Metadata Requirements

Every retrieval chunk should retain enough metadata for:

- filtering
- traceability
- source citation
- debugging
- answer evaluation

Important fields include:

```text
chunk_id
parent_id
document_id
document_type
chunk_type
question_id
chapter
section
subsection
content
page_start
page_end
source
```

Potential additional metadata:

```text
language
subject
class/grade
topic
element_id
image_reference
table_reference
formula_reference
```

---

# 14. Original PDF vs Markdown/JSON

The distinction should remain explicit.

## Markdown / JSON

Used for:

- parsing
- structure detection
- Q&A extraction
- chunking
- embeddings
- retrieval

## Original PDF

Used for:

- source verification
- page-level evidence
- image/diagram lookup
- layout verification
- provenance
- final citation/reference

Therefore, we should NOT independently create a completely separate PDF chunking pipeline unless later evaluation shows that the structured MinerU representation is insufficient for a particular use case.

---

# 15. Chunk Size Strategy

There should NOT be one global:

```python
chunk_size = 500
chunk_overlap = 100
```

for all content.

### Book

Start around:

```text
300–800 tokens
```

but allow structure/semantic boundaries to override the exact size.

### Answer Key

Preferred:

```text
1 Question = 1 Parent Q&A
```

Split only if the answer is too large.

### Student Answer

Preferred:

```text
1 Question = 1 Parent Student Answer
```

Split only when necessary.

---

# 16. Overlap Strategy

## Book

Some overlap may be useful for semantic continuity.

Example:

```text
Chunk 1: A B C D E F
Chunk 2:       E F G H I J
```

But overlap should not be the primary mechanism for preserving context.

Hierarchy metadata should do most of that work.

## Q&A

Do NOT blindly overlap across questions.

Avoid:

```text
Q1 + end of Q1 + Q2
```

because this can contaminate retrieval.

Prefer:

```text
Q1 → independent parent
Q2 → independent parent
Q3 → independent parent
```

---

# 17. Vector Database Organization

The data should be logically separated.

For example:

```text
Pinecone
 |
 +-- book_chunks
 |
 +-- answer_key_chunks
 |
 +-- student_answer_chunks
```

The exact Pinecone implementation can use namespaces or another equivalent logical separation.

The important principle is:

**Do not allow reference answers and student answers to become indistinguishable retrieval sources.**

---

# 18. Planned Retrieval Flow

For a student question:

```text
Student Q12
     |
     +----------------------+
     |                      |
     v                      v
Correct Answer             Book
     |                      |
     |                Relevant Concepts
     |                      |
     +----------+-----------+
                |
                v
         Evaluation Agent
                |
                v
       Score + Explanation
       Missing Points
       Mistakes
       Suggestions
```

The evaluation agent should receive grounded evidence rather than relying only on its internal knowledge.

---

# 19. Expected Evaluation Context

For a particular question, the eventual evaluation prompt/context should conceptually contain:

```text
QUESTION
+
STUDENT ANSWER
+
REFERENCE/CORRECT ANSWER
+
RELEVANT BOOK CONTENT
+
OPTIONAL FORMULA/FIGURE/TABLE EVIDENCE
```

Then the evaluation layer can determine:

- correctness
- completeness
- missing concepts
- factual errors
- formula/equation correctness
- reasoning quality
- suggested improvements

The exact scoring rubric is a separate design task.

---

# 20. Current Architecture in One View

```text
                        ORIGINAL PDF
                             |
                             v
                           MinerU
                             |
                 +-----------+-----------+
                 |                       |
                 v                       v
             Markdown                  JSON
                 |                       |
                 +-----------+-----------+
                             |
                             v
                    Structure Parser
                             |
                             v
                    Document Classifier
                             |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
        BOOK             ANSWER KEY       STUDENT PAPER
          |                  |                  |
          v                  v                  v
    Heading-aware        Question-aware    Question-aware
    Semantic chunks      Parent Q&A        Parent Student
          |                  |                  |
          +------------------+------------------+
                             |
                             v
                       Typed Chunks
                             |
                             v
                         Metadata
                             |
                             v
                        Embeddings
                             |
                             v
                         Vector DB
                             |
                             v
                         Retrieval
                             |
                             v
                    Answer Review Agent
                             |
                             v
                   Score + Feedback
```

---

# 21. What Is Already Decided

These decisions can currently be treated as the working architecture:

- MinerU is the extraction layer.
- MinerU Markdown + JSON are the canonical structured representations.
- Original PDF is retained for provenance/verification.
- There is no single generic chunking strategy.
- Book uses heading/hierarchy-aware semantic chunking.
- Answer key uses question-aware hierarchical chunking.
- Student answer sheet uses question-aware hierarchical chunking.
- Parent-child relationships are preserved.
- Q&A detection must use multiple signals, not one keyword.
- Formulas, tables and figures should be typed/preserved rather than blindly split.
- Metadata is a first-class part of every chunk.
- Book/reference/student data should remain logically separated during retrieval.

---

# 22. What Is Completed & Current Feature Status

*(Updated October 2026)*

### A. Canonical Chunk JSON Schema — ✅ DONE
**Implemented** as a `Chunk` dataclass with fields: `chunk_id`, `parent_id`, `document_id`, `document_type`, `chunk_type`, `content`, `chapter`, `section`, `subsection`, `question_id`, `page_start`, `page_end`, `source`, `language`, `subject`, `formula_reference`, `image_reference`, `table_reference`.

### B. Exact Chunking Algorithm — ✅ DONE (v2)
- Heading-aware hierarchy tracking for textbooks
- Q&A boundary detection for answer keys and student papers
- Token size target: 300–800 tokens with paragraph splits
- Parent-child relationship tracking and metadata flattening for Pinecone

### C. Question Detection — ✅ DONE (v2)
- ✅ Numbering patterns (`1.`, `1)`, `(1)`)
- ✅ Heading patterns (`## N Question N: ...`, `### Q1 ...`)
- ✅ Keyword markers (`Q`, `Question`, `Que`)
- ✅ Markdown formatting prefixes (`**1. ...**`, `*1. ...*`, `__1. ...__`)
- ⬜ Multilingual question markers (Hindi / regional) — IN PROGRESS
- ⬜ Nested subquestions `(a)`, `(b)`, `(c)` — NEXT
- ⬜ Automated MCQ options grouping — NEXT

### D. Retrieval & Reranking Strategy — ✅ DONE (v2)
- ✅ Dense retrieval via `all-MiniLM-L6-v2` + Pinecone Serverless
- ✅ Namespace-based logical separation (`book_chunks`, `answer_key_chunks`, `student_answer_chunks`)
- ✅ Cosine similarity thresholding (`SIMILARITY_THRESHOLD = 0.30`) to discard out-of-domain noise
- ✅ **Cross-Encoder Reranking** — Implemented with `cross-encoder/ms-marco-MiniLM-L-6-v2` (`rerank()`)
- ✅ Page number and chapter/section metadata preservation for grounded citations
- ⬜ Hybrid BM25 + Dense search — NEXT

### E. Answer Review Agent & LLM Fallback Architecture — ✅ DONE (v2)
- ✅ Grounded prompt with structured rubric (Score, Correctness, Completeness, Missing Points, Mistakes, Suggestions, Citations)
- ✅ Reference answer comparison + Textbook context integration
- ✅ **Multi-Model Auto-Fallback Engine**:
  - **Primary:** Google Gemini (`gemini-3-flash-preview` / `gemini-2.5-flash`)
  - **Fallback 1:** Groq (`openai/gpt-oss-120b` / `llama-3.1-8b-instant`)
  - **Fallback 2:** Local Ollama (`qwen3:4b`)
- ✅ **Anti-Hallucination & OOD Guardrails**:
  - Hard similarity cutoff filtering before LLM prompt assembly
  - System prompt guardrails instructing LLM not to fabricate missing syllabus criteria
  - Graceful degradation when question is out-of-syllabus

### F. Formula Rendering & PDF Assessment Reports — ✅ DONE (NEW — October 2026)
- ✅ **Formula Rendering Engine**:
  - Intercepts display formulas (`$$...$$`, `\[...\]`) and inline formulas (`$...$`, `\(...\)`)
  - Normalizes LaTeX (`\operatorname`, `\Big`, `\!`, `\begin{array}`) for mathtext compatibility
  - Generates transparent, 220 DPI PNG equation images cached in `formula_cache/`
  - Embeds equations as centered ReportLab flowables or inline text images
- ✅ **Unicode & Special Character Support**:
  - `DejaVuSans` TrueType font integration across all PDF styles
  - Eliminates missing-glyph black squares (`■`) for degrees (`°`), roots (`√`), Greek letters (`θ`, `Δ`), and exponents
  - Normalizes OCR ring operators (`\u2218`) and strips zero-width spaces (`\u200b`)
- ✅ **Student Assessment PDF Generators**:
  - Single-question review report (`generate_review_pdf`)
  - Comprehensive multi-question exam report with scorecards and performance bands (`generate_full_exam_pdf`)

---

# 23. Immediate Next Steps

The focus is now on **end-to-end integration and frontend delivery**:

```text
EXPAND QA INGESTION & TEST DIVERSE EXAM PAPERS
                    ↓
ENHANCE OCR PIPELINE FOR HANDWRITTEN PAPERS
                    ↓
SUBQUESTION / MCQ HANDLING
                    ↓
UI / GRADIO & STREAMLIT WEB APP
```

---

# 24. Current Project Position

*(Updated October 2026)*

```text
PDF extraction           → DONE (MinerU)
         |
         v
Structured output        → DONE (MD + JSON)
         |
         v
Chunking & Question v2   → DONE (Heading-aware + Markdown Bold support)
         |
         v
Vector DB & Namespace    → DONE (Pinecone Serverless)
         |
         v
Retrieval & Reranking    → DONE (Cosine Threshold + Cross-Encoder)
         |
         v
Answer Review Agent v2   → DONE (Gemini Primary + Groq Fallback + Ollama Fallback)
         |
         v
Guardrails & Prevention  → DONE (Threshold cutoff + Grounded negative constraints)
         |
         v
Formula Engine & PDFs    → DONE (Mathtext rendering + ReportLab Unicode + Clean PDFs)
         |
         v
Hardening & Edge Cases   → ACTIVE PHASE  ← WE ARE HERE
         |
         v
Interactive Web UI       → UPCOMING PHASE
```

**Implementation Notebook:** [`src/student_test_review_pipeline_poc/gradeforge_rag_pipeline.ipynb`](file:///c:/Users/IA%20Dev86/Nilesh/student_test_review_pipeline_POC/src/student_test_review_pipeline_poc/gradeforge_rag_pipeline.ipynb)
