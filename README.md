# OpenThaiRAG

OpenThaiRAG is an open-source Retrieval-Augmented Generation (RAG) framework designed specifically for Thai language processing. This project combines the power of vector databases, large language models, and information retrieval techniques to provide accurate and context-aware responses to user queries in Thai.

## Maintainer

Kobkrit Viriyayudhakorn (kobkrit@aieat.or.th), OpenThaiGPT Team.

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

## Installation

To install and run OpenThaiRAG using Docker Compose, follow these steps:

1. Ensure you have Docker and Docker Compose installed on your system.

2. Clone the OpenThaiRAG repository:
   ```
   git clone https://github.com/OpenThaiGPT/openthairag
   cd openthairag
   ```

3. Build and start the containers using Docker Compose:
   ```
   docker-compose up -d
   ```

   This command will:
   - Build the web service container
   - Start the Milvus standalone server
   - Start the etcd service
   - Start the MinIO service
   - Link all services together as defined in the docker-compose.yml file

4. Once all containers are up and running, the OpenThaiRAG API will be available at `http://localhost:5000`.

5. To stop the services, run:
   ```
   docker-compose down
   ```

Note: Ensure that port 5000 is available on your host machine, as it's used to expose the web service. Also, verify that you have sufficient disk space for the Milvus, etcd, and MinIO data volumes.

For production deployments, it's recommended to adjust the environment variables and security settings in the docker-compose.yml file according to your specific requirements.

## Containers

OpenThaiRAG utilizes several containers to provide its functionality. Here's an explanation of each container's role and purpose:

1. **web**:
   - Role: Main application container
   - Purpose: Hosts the Flask web service that provides the RESTful API for OpenThaiRAG. It handles document indexing, query processing, and interaction with other services.

2. **milvus**:
   - Role: Vector database
   - Purpose: Stores and manages document embeddings for efficient similarity search. It's crucial for the retrieval component of the RAG system.

3. **etcd**:
   - Role: Distributed key-value store
   - Purpose: Used by Milvus for metadata storage and cluster coordination. It ensures data consistency and helps manage the distributed nature of Milvus.

4. **minio**:
   - Role: Object storage
   - Purpose: Provides S3-compatible object storage for Milvus. It's used to store large objects and files that are part of the Milvus ecosystem.

These containers work together to create a robust and scalable infrastructure for the OpenThaiRAG system:

- The web container interacts with Milvus for vector operations.
- Milvus uses etcd for metadata management and MinIO for object storage.
- This architecture allows for efficient document embedding storage, retrieval, and query processing, which are essential for the RAG (Retrieval-Augmented Generation) functionality of OpenThaiRAG.

## API Documentation

For detailed API documentation and examples, please refer to our Postman collection:
[OpenThaiRAG API Postman Collection](https://universal-capsule-630444.postman.co/workspace/Travel-LLM~43ad4794-de74-4579-bf8f-24dbe26da1e5/collection/5145656-81239b64-fc7e-4f61-acfd-8e5916e037ce?action=share&creator=5145656)

## License

Apache 2.0
