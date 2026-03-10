# RAG Domain Expert Chatbot

This project builds a production-ready RAG chatbot with vector database,
conversation memory, and source citations.

## Architecture

User Query
    │
    ▼
Streamlit UI (app.py)
    │
    ▼
RAGChain (rag_chain.py)
    │
    ├── Conversation Memory
    ├── Query Expansion
    │
    ▼
Hybrid Retriever (retriever.py)
    │
    ├── Semantic Vector Search
    └── Keyword Search
    │
    ▼
Top-K Document Chunks
    │
    ▼
Prompt Construction
    │
    ▼
Open Ai
    │
    ▼
Answer + Source Citations

## Features

- Vector Database (Chroma)
- Retrieval-Augmented Generation
- Conversation Memory
- Source Citations
- Re-ranking

## Installation

pip install -r requirements.txt

## Domain

NASA space research and technical publications

## Dataset

- 60 NASA public PDF documents
- Raw files stored in `data/raw/`
- Source list stored in `docs/source_list.csv`

## Challenges

1. Slow Document Ingestion
One of the main engineering challenges was the long ingestion time required to process the document corpus.
During testing, ingestion required significant time because:
- the system processes 60 large NASA reports
- each document produces 200–500 text chunks
- embeddings must be generated for every chunk
- embeddings are stored in the vector database
As a result, the ingestion step may take several minutes depending on hardware.

2. Handling Technical Document Structure
NASA reports contain complex formatting including:
- multi-column layouts
- tables
- figures and captions
Basic PDF parsers often produce noisy text extraction.

3. Retrieval Accuracy for Technical Content
Technical documents often include:
- mission acronyms
- specific spacecraft names
- numerical specifications
Pure semantic search sometimes failed to retrieve relevant passages.

4. Follow-Up Question Understanding
Users frequently ask follow-up questions referring to previous answers.
Without context, the retriever may misinterpret queries.
Example:
 ``` 
    User:
    “What is the Artemis program?”

    Follow-up:
    “When will it launch?”
 ```

## Results
The final system successfully demonstrates a complete domain-specific RAG chatbot pipeline.
Key outcomes include:

### Document Processing
- ~60 NASA technical reports processed
- thousands of text chunks embedded
- embeddings stored in a persistent vector database

### Retrieval Performance
The hybrid retrieval system consistently identifies relevant document passages.
Testing shows that the system reliably retrieves information about:
- NASA missions
- Mars exploration research
- Artemis program details
- spacecraft technology

### Conversational Interaction
The chatbot supports:
- multi-turn conversations
- follow-up questions
- contextual query interpretation
Conversation memory allows the system to maintain topic continuity.

### Source Grounding
Generated answers include explicit source citations, improving transparency and reliability.
Example citation format:

 ```
    [Source 2] NASA Technical Report (Page 15)
 ```

This enables users to verify information directly from the original documents.

### System Capabilities

The completed system demonstrates:
- document ingestion and embedding
- hybrid semantic retrieval
- grounded LLM generation
- conversational memory
- interactive UI

Together, these components illustrate how RAG architecture can convert large document collections into an interactive domain expert assistant.