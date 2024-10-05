import os
import requests
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_and_index_docs(docs_dir='./docs', index_url='http://localhost:5000/index'):
    """
    Read .txt files from the specified directory and index their contents.
    
    :param docs_dir: Directory containing the .txt files
    :param index_url: URL of the indexing endpoint
    """
    indexed_files = 0
    non_indexed_files = 0
    
    for filename in os.listdir(docs_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(docs_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                # Split content into chunks of max 1000 characters, including the title in each chunk
                chunks = []
                lines = content.split('\n')
                title = lines[0]  # Get the first line as the title
                current_chunk = title + '\n'  # Start each chunk with the title
                for line in lines[1:]:  # Skip the first line (title) in this loop
                    if len(current_chunk) + len(line) + 1 <= 1000:  # +1 for newline
                        current_chunk += line + '\n'
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = title + '\n' + line + '\n'  # Start a new chunk with the title
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Index each chunk separately
                file_indexed = True
                for chunk in chunks:
                    # Prepare the data for indexing
                    data = {
                        "text": chunk
                    }
                    
                    # Send the content to be indexed
                    response = requests.post(index_url, json=data)
                    
                    if response.status_code == 201:
                        logger.info(f"Successfully indexed chunk from {filename}")
                    else:
                        logger.error(f"Failed to index chunk from {filename}. Status code: {response.status_code}")
                        logger.error(f"Response: {response.text}")
                        file_indexed = False
                
                if file_indexed:
                    indexed_files += 1
                    logger.info(f"Finished indexing all chunks from {filename}")
                else:
                    non_indexed_files += 1
            
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                non_indexed_files += 1

    logger.info(f"Indexing complete. Indexed files: {indexed_files}, Non-indexed files: {non_indexed_files}")

if __name__ == "__main__":
    read_and_index_docs()
