# README for RAG Agent

## Overview

The RAG agent is designed to:
   1. Index PDF documents and store them in a vector database (`Chroma`).
   2.	Retrieve relevant documents based on user queries (`langchain`).
   3.	Generate responses using a language model (`Ollama`) based on the retrieved documents.

## Project Structure
   ```bash
.
├── indexing.py               # Script to index documents and store them in Chroma
├── query_data.py             # Script to query the indexed documents
├── get_embedding_function.py  # Function to retrieve embeddings for text
├── app.py                     # Streamlit app to interact with the RAG agent
└── .env                       # Environment variables (e.g., CHROMA_PATH)
   ```

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Setup Instructions](#setup-instructions)
4. [Running the System](#running-the-system)
5. [Components](#Components)
6. [Version 2.0](#Version-2.0)

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:

-  Python 3.7 or higher
-  pip (Python package installer)
-  Virtual environment (optional but recommended)
-   Ollama installed (follow the [Ollama installation guide](https://ollama.com/docs/installation) if you haven't installed it yet)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>

2. **Create a virtual environment (optional)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. **Install required packages**:
   ```bash
    pip install -r requirements.txt

## Setup Instructions

1. **Environment Variables**: Create a `.env` file in the root directory of your project and set the following environment variable:
   ```bash
   CHROMA_PATH="./chroma_db"  # or any path where you want to store the Chroma database

2. **Create a Data Directory**: Ensure you have a directory named `data` in the root of your project where the PDF files will be uploaded and indexed.

3. **Pull Ollama Models**: 
   To use the required models, run the following commands to pull the `llama3` and `nomic-embed-text` models:
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
4. 	**Start the Ollama Local Server**:
   To use the Ollama model, you need to start the Ollama local server. Open a terminal and run:
      ```bash
      ollama serve
   This command will start the Ollama server, allowing the RAG agent to access the required models.


## Running the System

To run the RAG agent, execute the following command in your terminal:

   ```bash 
  streamlit run app.py
   ```
This command will start a Streamlit server, and you will see output indicating the local URL where the app is hosted.


## Components

1. **Document Indexing** (`indexing.py`) <br>
   The indexing.py script is responsible for loading PDF documents, splitting them into manageable chunks, and storing them in the Chroma vector database.
	- Key Functions:
      - `retrieve_documents(data)`: Loads PDF documents from the specified directory. 
      - `partition_documents(documents)`: Splits documents into chunks using `RecursiveCharacterTextSplitter`. 
      - `store_in_chroma(chunks)`: Stores the chunks in the Chroma database, checking for duplicates. 
      - `clear_database()`: Clears the Chroma database if the `--reset` flag is used.
   - Usage:
        ```bash
        python indexing.py --reset  # To reset the database and re-index documents
        python indexing.py           # To index documents without resetting
        ```
2. **Document Querying** (`query_data.py`) <br>
   The `query_data.py` script handles user queries, retrieves relevant document chunks from Chroma, and generates responses using a language model.
    - Key Functions:
      - `def query_rag(query_text)`: Retrieves relevant chunks based on the query and generates a response using the Ollama model.

   - Usage:
        ```bash
        python query_data.py "Your query text here"
        ```
3.  **Embedding Retrieval** (`get_embedding_function.py`) <br>
      This module defines a function to retrieve embeddings for text using the Ollama embeddings model.
    - Key Functions:
      - `def get_embedding_function(text)`: Returns an instance of the `OllamaEmbeddings` model for text embedding.


## Version 2.0
   **Overview**   
   As per the requirements, I was trying to build the Hierarchical Tree-based Indexing, BM25, query expansion and re-ranking. However, I was unable to automate the indexing of the book. While attempting to implement the hierarchical tree-based indexing using the common methods, every book would have to be preprocessed manually, and I was unable to do it automatically. But if someone had been able to automate that, here's my approach utilize the data efficiently.
   
   - **Adding data to index**
     ```python
      def index_node(self, current_node, source_identifier):
        # Base case: if the current node is None, exit the function
        if current_node is None:
            return
    
        # Generate a unique identifier for the node
        chunk_id = f"{source_identifier}/{current_node.title}"
        node_content = current_node.content if current_node.content else ""
    
        # Generate embeddings using the models
        node_embedding = self.model.encode(node_content)
    
        # Save node information to the index
        self.index[chunk_id] = {
            'title': current_node.title,
            'page': current_node.page,
            'content': node_content,
            'source': source_identifier
        }
    
        # Add the document and its embeddings to ChromaDB
        self.collection.add(
            documents=[node_content],
            metadatas=[{'key': chunk_id}],
            embeddings=[node_embedding],
            ids=[chunk_id]  
        )
    
        # Add the content to the corpus list
        self.corpus.append(node_content)

   - Re-Ranking and searching in the db
       ```python
        class EnhancedSearch:
            def __init__(self, vector_db, docs, bm25_corpus):
                self._vector_database = vector_db
                self._vector_database.add_documents(docs)
                self._bm25_collection = bm25_corpus
                
                tokenized_docs = [document.split(" ") for document in bm25_corpus]
                self._bm25_model = BM25Okapi(tokenized_docs)
                
            def perform_vector_search(self, query_text: str, top_k=3) -> Dict[str, float]:
                results = {}
                retrieved_docs = self._vector_database.similarity_search_with_relevance_scores(query=query_text, k=top_k)
                
                for document, score in retrieved_docs:
                    results[document.page_content] = score
                    
                return dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
            
            def perform_bm25_search(self, query_text: str, top_k=3) -> Dict[str, float]:
                tokenized_query = query_text.split(" ")
                scores = self._bm25_model.get_scores(tokenized_query)
                documents_with_scores = dict(zip(self._bm25_collection, scores))
                sorted_documents = sorted(documents_with_scores.items(), key=lambda item: item[1], reverse=True)
                
                return dict(sorted_documents[:top_k])
                
            def merge_search_results(self, vector_results: Dict[str, float], 
                                     bm25_results: Dict[str, float]) -> Dict[str, float]:
                
                def scale_scores(score_dict):
                    epsilon = 0.05
                    min_score = min(score_dict.values())
                    max_score = max(score_dict.values())
                    low, high = 0.05, 1
                    
                    if max_score == min_score:
                        return {key: high if max_score > 0.5 else low for key in score_dict.keys()}
            
                    return {key: low + ((value - min_score) / (max_score - min_score)) * (high - low) for key, value in score_dict.items()}
                
                scaled_vector_results = scale_scores(vector_results)
                scaled_bm25_results = scale_scores(bm25_results)
        
                combined_results = {}
                for key, value in scaled_vector_results.items():
                    combined_results[key] = value
        
                for key, value in scaled_bm25_results.items():
                    if key in combined_results:
                        combined_results[key] = max(combined_results[key], value)
                    else:
                        combined_results[key] = value
        
                return combined_results
        
            def search_documents(self, query_text: str, top_k=3, include_bm25=True) -> Dict[str, float]:
                vector_results = self.perform_vector_search(query_text, top_k=top_k)
                
                if include_bm25:
                    bm25_results = self.perform_bm25_search(query_text, top_k=top_k)
                    if bm25_results:
                        merged_results = self.merge_search_results(vector_results, bm25_results)
                        return dict(sorted(merged_results.items(), key=lambda item: item[1], reverse=True))
                
                return vector_results

## Postscript

This is my first attempt at building a Retrieval-Augmented Generation (RAG) model. Throughout this project, I have been learning and developing the system side by side. The process has been both challenging and rewarding, as I've had to delve into various concepts such as document indexing, vector databases, and language model integration.

I appreciate the resources and community support that have guided me along the way. This project not only represents a technical achievement but also a significant step in my journey to understand and implement advanced AI techniques. I look forward to further refining this model and exploring its potential applications.


