from flask import Flask, request, jsonify
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, utility, DataType
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import os
import numpy as np
import logging
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a StreamHandler to output logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)

logger.info("Logger initialized for Flask application")


# Configuration for Milvus and vLLM hosts
MILVUS_HOST = os.environ.get('MILVUS_HOST', 'milvus')
MILVUS_PORT = os.environ.get('MILVUS_PORT', '19530')
VLLM_HOST = os.environ.get('VLLM_HOST', '172.17.0.1:8000')

# Update Milvus connection settings
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

# Function to initialize Milvus collection
def initialize_milvus_collection():
    # Check if collection exists
    if not utility.has_collection("document_embeddings"):
        # Create collection if it doesn't exist
        # You may need to adjust the schema based on your specific requirements

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),  # Adjust dim if needed
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
        ]
        schema = CollectionSchema(fields, "Document embeddings for travel information")
        collection = Collection("document_embeddings", schema)
        
        # Create an IVF_FLAT index for the embedding field
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index("embedding", index_params)
    else:
        collection = Collection("document_embeddings")
    
    return collection

# Initialize Milvus collection
collection = initialize_milvus_collection()

# Flask app setup
app = Flask(__name__)

# Load BAAI/bge-m3 model and tokenizer
bge_model = AutoModel.from_pretrained("BAAI/bge-m3")
bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

# Function to generate embeddings
def generate_embedding(text):
    inputs = bge_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = bge_model(**inputs).pooler_output
    return embeddings

# Rerank documents based on cosine similarity
def rerank_documents(query_embedding, document_embeddings):
    # Ensure query_embedding is a 2D array
    if isinstance(query_embedding, list):
        query_embedding = np.array(query_embedding).reshape(1, -1)
    elif isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.reshape(1, -1)
    
    # Ensure document_embeddings is a 2D array
    if isinstance(document_embeddings, list):
        document_embeddings = np.array(document_embeddings)
    if len(document_embeddings.shape) == 1:
        document_embeddings = document_embeddings.reshape(1, -1)
    
    # Check if document_embeddings is empty
    if document_embeddings.size == 0:
        logging.warning("Document embeddings array is empty")
        return []
    
    logger.info(f"Query embedding shape: {query_embedding.shape}, Document embeddings shape: {document_embeddings.shape}")
    logger.info(f"Query embedding: {query_embedding}, Document embeddings: {document_embeddings}")
    
    similarities = cosine_similarity(query_embedding, document_embeddings)
    ranked_documents = sorted(enumerate(similarities.flatten()), key=lambda x: x[1], reverse=True)
    return ranked_documents

# Function to generate text using vllm
def generate_vllm_response(prompt):
    url = f'http://{VLLM_HOST}/v1/completions'
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": ".",
        "prompt": f"<|im_start|>system\nคุณคือผู้ช่วยตอบคำถามที่ฉลาดและซื่อสัตย์<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 40,
        "stop": ["<|im_end|>"]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()['choices'][0]['text']

# Flask route for index page
@app.route("/", methods=["GET"])
def index():
    return "Hello", 200

# Flask route for indexing text
@app.route("/index", methods=["POST"])
def index_text():
    try:
        # Get text from request
        data = request.get_json()
        text = data.get("text")
        
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Generate embedding for the text
        embedding = generate_embedding(text).numpy().flatten().tolist()

        # Prepare the entity to be inserted
        entity = {
            "text": text,
            "embedding": embedding
        }

        # Log all parameters of the entity
        logger.info("Indexing new document:")
        logger.info(f"Text: {text[:100]}...")  # Log first 100 characters of text
        logger.info(f"Embedding shape: {np.array(embedding).shape}")
        logger.info(f"Embedding sample: {embedding[:5]}...")  # Log first 5 elements of embedding
        # Log the entire entity
        logger.info(f"Full entity: {entity}")

        # Insert the entity into Milvus
        insert_result = collection.insert([entity])

        # Log the insert result
        logger.info(f"Insert result: {insert_result}")

        # Ensure the changes are immediately searchable
        collection.flush()

        return jsonify({
            "message": "Text indexed successfully",
            "id": insert_result.primary_keys[0]
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask route for deleting all indexed documents
@app.route("/delete", methods=["DELETE"])
def delete_all_documents():
    try:
        # Delete all entities in the collection
        delete_result = collection.delete(expr="id >= 0")  # Use a condition that matches all documents
        
        # Log the delete result
        logger.info(f"Delete result: {delete_result}")

        # Ensure the changes are immediately reflected
        collection.flush()

        return jsonify({
            "message": "All documents deleted successfully",
            "num_deleted": delete_result.delete_count
        }), 200

    except Exception as e:
        logger.error(f"Error deleting documents: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
# Flask route for listing all indexed documents
@app.route("/list", methods=["GET"])
def list_all_documents():
    try:
        # Query all entities in the collection
        results = collection.query(
            expr="",
            output_fields=["id", "text", "embedding"],
            limit=collection.num_entities
        )
        
        # Prepare the response
        documents = [
            {
                "id": str(doc['id']),
                "text": doc['text'][:100] + "...",
                "embedding": [float(x) for x in doc['embedding'][:10]]  # Convert to list of floats
            } for doc in results
        ]
        # Log the number of documents retrieved
        logger.info(f"Retrieved {len(documents)} documents")

        return jsonify({
            "message": "Documents retrieved successfully",
            "documents": documents
        }), 200

    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Flask route for handling user queries
@app.route("/query", methods=["POST"])
def query():
    # Get user query from request
    data = request.get_json()
    query_text = data["query"]

    # Step 1: Generate query embedding
    query_embedding = generate_embedding(query_text).numpy().flatten().tolist()
    # Prepare search parameters
    search_param = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
        
    # Step 2: Retrieve top-10 documents from Milvus
    search_results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_param,
        limit=10,
        output_fields=["id", "text", "embedding"],
        expr=None
    )
    # Extract document texts and embeddings
    retrieved_documents = []
    document_embeddings = []
    for hits in search_results:
        for hit in hits:
            logger.info(f"Retrieved document: {hit.entity.get('text')[:50]}...")
            retrieved_documents.append(hit.entity)
            embedding = hit.entity.get('embedding')
            logger.info(f"Retrieved embedding: {embedding[:5]}... (truncated)")
            if embedding is not None:
                document_embeddings.append(embedding)

    # Log retrieved documents and document embeddings
    logger.info(f"Retrieved documents: {retrieved_documents}")
    logger.info(f"Document embeddings: {document_embeddings}")

    # Step 3: Re-rank retrieved documents using BGE reranker
    ranked_indices = rerank_documents(query_embedding, document_embeddings)
    top_documents = [retrieved_documents[i] for i, _ in ranked_indices[:3]]

    # Step 4: Prepare prompt for VLLM with top re-ranked documents
    prompt = f"Based on the following documents, answer the query:\n\n"
    prompt += "\n\n".join([doc.get('text') for doc in top_documents])
    prompt += f"\n\nQuery: {query_text}"

    # Step 5: Generate final response with VLLM
    output = generate_vllm_response(prompt)

    # Return response to the client
    return jsonify({"query": query_text, "response": output})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)