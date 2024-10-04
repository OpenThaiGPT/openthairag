# OpenThaiRAG
OpenThaiRAG is an open-source Retrieval-Augmented Generation (RAG) framework designed specifically for Thai language processing. This project combines the power of vector databases, large language models, and information retrieval techniques to provide accurate and context-aware responses to user queries in Thai.

## Maintainer
Kobkrit Viriyayudhakorn (kobkrit@aieat.or.th), OpenThaiGPT Team.

## Postman
https://universal-capsule-630444.postman.co/workspace/Travel-LLM~43ad4794-de74-4579-bf8f-24dbe26da1e5/collection/5145656-81239b64-fc7e-4f61-acfd-8e5916e037ce?action=share&creator=5145656

## License
Apache 2.0

## Key Features

- **Vector Database Integration**: Utilizes Milvus for efficient storage and retrieval of document embeddings.
- **Multilingual Embedding Model**: Incorporates the BAAI/bge-m3 model for generating high-quality embeddings for Thai text.
- **Advanced Retrieval**: Implements a two-stage retrieval process with initial vector search and subsequent re-ranking for improved accuracy.
- **Large Language Model Integration**: Seamlessly integrates with vLLM for generating human-like responses based on retrieved context.
- **RESTful API**: Offers a Flask-based web API for easy integration into various applications.

## Core Functionalities

1. **Document Indexing**: Allows users to index Thai documents, generating and storing embeddings for efficient retrieval.
2. **Query Processing**: Handles user queries by finding relevant documents and generating context-aware responses.
3. **Document Management**: Provides endpoints for listing and deleting indexed documents.

OpenThaiRAG aims to enhance natural language understanding and generation for Thai language applications, making it a valuable tool for developers working on chatbots, question-answering systems, and other NLP projects focused on Thai language processing.
