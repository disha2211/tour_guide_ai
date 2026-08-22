# 🗺️ Tour Guide AI

> A multimodal AI tourism assistant that identifies landmarks from images and answers contextual questions using image similarity, semantic retrieval, and Retrieval-Augmented Generation (RAG).

## 📌 Overview

I built **Tour Guide AI** as a multimodal question-answering system that combines **Computer Vision, Vector Search, RAG, and Large Language Models** to create an interactive tourism assistant.

Instead of requiring users to know the name of a landmark beforehand, the application accepts an image as input, identifies the most visually similar landmark from its knowledge base, and then allows the user to ask natural-language questions about it.

For example:

**Input:** A photograph of Prague Castle  
**→ Image similarity search**  
**→ Identifies `prague_castle`**  
**→ Retrieves relevant historical information**  
**→ Gemini generates a grounded response**

The system therefore follows:

```text
User Image
    ↓
Image Embedding
    ↓
ChromaDB Similarity Search
    ↓
Landmark Identification
    ↓
User Question
    ↓
Semantic Text Retrieval
    ↓
Relevant Knowledge Chunks
    ↓
Gemini 2.5 Flash
    ↓
Context-Grounded Answer
```

---

## ✨ Features

### 🖼️ Image-Based Landmark Identification

Users can upload a `.jpg`, `.jpeg`, or `.png` image through the Streamlit interface.

The image is converted into an embedding using **Img2Vec**, and the embedding is compared against the landmark image collection stored in ChromaDB.

The nearest image determines the predicted landmark category.

---

### 🔎 Semantic Knowledge Retrieval

Once the landmark is identified, the application searches the corresponding textual knowledge collection.

Instead of relying on keyword matching, the question is converted into an embedding and compared against stored document chunks using semantic similarity.

The system retrieves the **top 3 relevant chunks** as context for answer generation.

---

### 🤖 Retrieval-Augmented Generation

The retrieved knowledge is passed to **Google Gemini 2.5 Flash**.

The prompt explicitly instructs the model to answer using only the supplied context, reducing the possibility of generating information unrelated to the project's knowledge base.

```text
Retrieved Context
       +
User Question
       ↓
Gemini 2.5 Flash
       ↓
Grounded Answer
```

---

### 🧠 Multimodal Retrieval Pipeline

The project combines two different retrieval modalities:

| Modality | Technique | Purpose |
|---|---|---|
| Image | Img2Vec embeddings | Identify landmark |
| Text | Gemini embeddings | Retrieve relevant information |
| Generation | Gemini 2.5 Flash | Produce final answer |

This allows the application to bridge **visual input → semantic knowledge → natural-language reasoning**.

---

### 💾 Persistent Vector Database

I used **ChromaDB PersistentClient** instead of an in-memory database.

This allows image and document embeddings to persist locally between application executions.

The database is stored under:

```text
./db
```

---

### 📚 Automatic Knowledge Base Construction

The `create_db.py` pipeline automatically processes the project's dataset.

For every tourism category, it:

1. Reads landmark images.
2. Adds them to the image collection.
3. Loads corresponding `.txt` knowledge files.
4. Splits documents into smaller chunks.
5. Generates semantic embeddings.
6. Stores the chunks and metadata in ChromaDB.

This separates **knowledge-base construction** from **runtime inference**.

---

## 🏗️ Architecture

The project follows a lightweight **two-stage retrieval architecture**.

### Stage 1 — Visual Retrieval

```text
                    ┌─────────────────┐
                    │   User Image    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │    OpenCV       │
                    │ Image Loading   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │     Img2Vec     │
                    │ Image Embedding │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │    ChromaDB     │
                    │ Image Collection│
                    └────────┬────────┘
                             ↓
                    Top-1 Similar Image
                             ↓
                    Landmark Category
```

### Stage 2 — Knowledge Retrieval + Generation

```text
User Question
      ↓
Gemini Text Embedding
      ↓
ChromaDB
documents_<landmark>
      ↓
Top-3 Relevant Chunks
      ↓
Context Construction
      ↓
Gemini 2.5 Flash
      ↓
Final Answer
```

---

## 🧩 System Components

### `main.py`

Acts as the **Streamlit application layer**.

Responsibilities:

- Provides the web interface.
- Accepts image uploads.
- Displays the uploaded image.
- Invokes landmark classification.
- Accepts natural-language questions.
- Triggers retrieval and generation.
- Displays the generated answer.

---

### `create_db.py`

Acts as the **knowledge ingestion pipeline**.

Responsibilities:

- Creates the persistent ChromaDB client.
- Creates the image collection.
- Ingests landmark images.
- Creates landmark-specific document collections.
- Loads `.txt` knowledge sources.
- Performs document chunking.
- Generates embeddings.
- Stores documents and metadata.

---

### `query_data.py`

Contains the **runtime AI pipeline**.

It implements three major operations:

```python
classify_img()
get_most_similar_chunks()
create_response()
```

#### `classify_img()`

Converts the uploaded image into an Img2Vec embedding and performs top-1 similarity search.

#### `get_most_similar_chunks()`

Queries the landmark-specific document collection and retrieves the top 3 semantically similar chunks.

#### `create_response()`

Constructs the RAG prompt and sends the retrieved context to Gemini 2.5 Flash.

---

### `utils.py`

Contains shared configuration and the custom image embedding implementation.

The `ImageEmbeddings` class integrates **Img2Vec** with ChromaDB's embedding-function interface.

This allows the image model to be used directly during vector database insertion and querying.

---

## 📁 Project Structure

```text
tour_guide_ai/
│
├── data/
│   ├── landmark_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── information.txt
│   │
│   ├── landmark_2/
│   │   ├── image1.jpg
│   │   └── information.txt
│   │
│   └── ...
│
├── db/
│   └── ChromaDB persistent storage
│
├── create_db.py
├── query_data.py
├── utils.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🔧 Technology Stack

### AI / ML

- **Img2Vec** — image representation and similarity search
- **Google Gemini Embeddings** — semantic document retrieval
- **Google Gemini 2.5 Flash** — contextual answer generation

### RAG / Vector Search

- **ChromaDB** — persistent vector database
- **LangChain** — document loading, chunking, and prompt construction

### Computer Vision

- **OpenCV** — image loading and preprocessing
- **PIL** — image conversion

### Application

- **Streamlit** — interactive web interface

### Configuration

- **python-dotenv** — environment variable management

---

# 🧠 Architectural Decisions

## 1. Vector Search Instead of Traditional Classification

I chose an **embedding-based image retrieval approach** rather than training a dedicated landmark classification model.

The system represents images as vectors and performs similarity search against the existing image knowledge base.

### Why?

- No custom classifier training pipeline is required.
- New landmarks can be added by inserting images into the vector database.
- The approach naturally supports similarity-based retrieval.
- It fits well with the project's RAG architecture.

---

## 2. ChromaDB as the Vector Store

I selected **ChromaDB** as the vector database because the project is designed as a lightweight local application.

Using `PersistentClient` provides:

- Local persistence
- Simple Python integration
- Native embedding-function support
- Straightforward similarity querying
- No external database infrastructure for development

This keeps the architecture simple while still implementing a genuine vector-search pipeline.

---

## 3. Separate Image and Text Collections

I deliberately separated visual and textual retrieval.

```text
imgs
 │
 └── Image embeddings

documents_<landmark>
 │
 └── Text embeddings
```

This prevents mixing different embedding spaces and allows each modality to use an embedding model appropriate for its data.

---

## 4. Landmark-Specific Knowledge Collections

Rather than searching the entire document corpus for every question, the application first identifies the landmark and then queries:

```text
documents_<identified_landmark>
```

This effectively performs **hierarchical retrieval**:

```text
Global Image Retrieval
        ↓
Landmark Identification
        ↓
Restricted Knowledge Retrieval
        ↓
Answer Generation
```

This reduces the search space and makes the retrieved context more domain-specific.

---

## 5. Chunking Before Embedding

Documents are split using `RecursiveCharacterTextSplitter` with:

```text
chunk_size   = 300
chunk_overlap = 100
```

The reasoning is to avoid embedding entire documents as a single vector.

Smaller overlapping chunks provide more granular semantic retrieval while retaining contextual continuity between neighboring chunks.

---

## 6. Top-K Retrieval Before Generation

The system retrieves only the most relevant three chunks:

```python
n_results=3
```

These chunks form the context supplied to Gemini.

This follows a standard RAG principle:

```text
Large Knowledge Base
       ↓
Semantic Retrieval
       ↓
Small Relevant Context
       ↓
LLM
```

rather than sending the entire knowledge base to the model.

---

## 7. Grounded Generation

The generation prompt explicitly constrains Gemini to answer from the retrieved context.

Conceptually:

```text
"Answer the question based only on the following context"
```

This establishes a basic grounding mechanism and reduces unnecessary reliance on the model's pretrained knowledge.

---

## 8. Persistent Local Architecture

The application uses:

```python
chromadb.PersistentClient(path=DB_PATH)
```

instead of an external vector database.

This was an intentional architectural trade-off:

**Advantages**

- Simple setup
- Low infrastructure overhead
- Easy local development
- Persistent embeddings

**Trade-off**

- Less suitable for large-scale multi-user production deployment.

---

# 🔄 End-to-End Data Flow

### Knowledge Base Creation

```text
Tourism Dataset
      ↓
┌───────────────┬─────────────────┐
│    Images     │    Text Files   │
└───────┬───────┴────────┬────────┘
        ↓                ↓
     Img2Vec       Recursive Chunking
        ↓                ↓
 Image Embeddings   Gemini Embeddings
        ↓                ↓
        └───────┬────────┘
                ↓
             ChromaDB
```

### User Query

```text
                User
                 │
                 ▼
          Upload Landmark Image
                 │
                 ▼
             Img2Vec
                 │
                 ▼
       ChromaDB Image Search
                 │
                 ▼
         Landmark Identification
                 │
                 ▼
           User Question
                 │
                 ▼
       Semantic Text Retrieval
                 │
                 ▼
          Top-3 Text Chunks
                 │
                 ▼
        Gemini 2.5 Flash
                 │
                 ▼
          Grounded Response
```

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone <repository-url>
cd tour_guide_ai
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Gemini API

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

The API key is loaded through `python-dotenv` and is not hardcoded into the application.

---

## 5. Prepare the Dataset

The expected structure is:

```text
data/
├── landmark_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── information.txt
│
├── landmark_2/
│   ├── image1.jpg
│   └── information.txt
│
└── ...
```

Each directory represents a landmark/category.

Images are used for visual retrieval, while `.txt` files provide the textual knowledge used by the RAG pipeline.

---

## 6. Build the Vector Database

Run:

```bash
python create_db.py
```

This creates the persistent ChromaDB collections under:

```text
./db
```

---

## 7. Launch the Application

```bash
streamlit run main.py
```

The Streamlit interface will allow you to:

1. Upload a landmark image.
2. View the identified landmark.
3. Ask a question.
4. Retrieve relevant knowledge.
5. Receive a Gemini-generated answer.

---

# 💡 Example Interaction

```text
User uploads:
prague_castle.jpg

↓

Detected:
Prague Castle

↓

User:
"When was it built?"

↓

Semantic Retrieval:
Top 3 relevant historical chunks

↓

Gemini 2.5 Flash:
Generates an answer using the retrieved context.
```

---

# 🔐 Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key |

The `.env` file should never be committed to version control.

---

# 📊 Current System Characteristics

| Component | Implementation |
|---|---|
| Image representation | Img2Vec |
| Image retrieval | ChromaDB |
| Text embeddings | Gemini Embeddings |
| Text retrieval | ChromaDB semantic search |
| Retrieved chunks | Top 3 |
| LLM | Gemini 2.5 Flash |
| Vector persistence | Local ChromaDB |
| Document splitting | Recursive Character Splitting |
| UI | Streamlit |
| Image processing | OpenCV + PIL |

---

# 🔮 Future Improvements

I would extend the project in the following directions:

### 1. Better Image Retrieval

Replace or augment Img2Vec with modern vision-language embeddings such as **CLIP** for stronger semantic image retrieval.

### 2. Hybrid Retrieval

Combine:

```text
Dense Vector Search
        +
Keyword/BM25 Search
        ↓
Reranking
        ↓
LLM
```

to improve retrieval for specific historical names, dates, and locations.

### 3. Source-Aware RAG

Expose retrieved documents and metadata directly in the UI:

```text
Answer
   ↓
Sources
   ├── Historical document
   ├── Tourism document
   └── Reference metadata
```

### 4. Retrieval Evaluation

Introduce measurable evaluation metrics such as:

- Recall@K
- MRR
- Precision@K
- Context relevance
- Faithfulness
- Answer correctness

### 5. Production Deployment

A production architecture could evolve toward:

```text
Streamlit / React
        ↓
FastAPI
        ↓
Retrieval Service
        ↓
Vector Database
        ↓
Gemini API
```

with the vector database moved from local ChromaDB to a managed/scalable solution.

---

# 🎯 What I Learned

Building this project helped me understand how multimodal AI systems can be constructed by combining specialized components rather than relying on a single model.

The main concepts I implemented were:

- Image embeddings
- Vector similarity search
- Semantic document retrieval
- Document chunking
- Retrieval-Augmented Generation
- Prompt grounding
- Persistent vector databases
- Multimodal retrieval pipelines
- LLM-based answer generation

The most important architectural idea was separating **visual identification** from **knowledge retrieval**:

```text
Vision → Identify the Entity
                 ↓
          Retrieve Knowledge
                 ↓
            LLM → Explain
```

This makes the system modular and allows each component to be improved independently.

---

# 👩‍💻 Project Focus

This project demonstrates my practical work with:

**Computer Vision + Embeddings + Vector Databases + RAG + LLMs + AI Application Development**

Rather than building a standalone chatbot, I focused on constructing an end-to-end AI pipeline where **unstructured visual input is converted into a grounded natural-language response**.
