import hashlib
import os
import requests
import mimetypes
import pandas as pd
import time



def upload_txt_files(folder_path, upload_url, num_files):
    i = 0
    for files in os.listdir(folder_path):
        _, ext = os.path.splitext(files)
        # Ingest only txt files
        if ext.lower() == ".txt":
            file_path = os.path.join(folder_path, files)
            print(upload_document(file_path, upload_url))
            i += 1
            if i > num_files:
                break


def upload_document(file_path, url):
    headers = {
        'accept': 'application/json'
    }
    mime_type, _ = mimetypes.guess_type(file_path)
    files = {
        'file': (file_path, open(file_path, 'rb'), mime_type)
    }
    response = requests.post(url, headers=headers, files=files)

    return response.text

def upload_pdf_files(folder_path, upload_url, num_files):
    i = 0
    for files in os.listdir(folder_path):
        _, ext = os.path.splitext(files)
        # Ingest only pdf files
        if ext.lower() == ".pdf":
            file_path = os.path.join(folder_path, files)
            print(upload_document(file_path, upload_url))
            i += 1
            if i > num_files:
                break

                import pandas as pd

# Load the JSONL file into a DataFrame
jsonl_file_path = '/home/kobkrit/data/tat/attractions_rag.jsonl'
df = pd.read_json(jsonl_file_path, lines=True)

# Print the DataFrame to verify
print(df.head())

# Example of df's row
# UUID                           e59504fe-895a-4747-b048-bbaad81f659e
# Created Date                                             2024-09-23
# Updated Date                                             2024-09-23
# Title                                     วัดดอนแก้ว (Wat Don Kaew)
# Content           มีพระพุทธรูปแกะสลักหินอ่อน เป็นปฏิมากรรมของพม่...
# URL                                                                
# Published Date              
                       
for index, row in df.iterrows():
    document = f"Title: {row['Title']}\nContent: {row['Content']}\nUUID: {row['UUID']}\nCreated Date: {row['Created Date']}\nUpdated Date: {row['Updated Date']}\nURL: {row['URL']}\nPublished Date: {row['Published Date']}"
    # Create the docs directory if it doesn't exist
    if not os.path.exists('./docs'):
        os.makedirs('./docs')
    
    # Define the file path using the title of the document
    short_title = row['Title'][:15].replace('/', '')  # Truncate title to 15 characters and remove "/"
    short_hash = hashlib.md5(row['Title'].encode()).hexdigest()[:5]  # Create a short hash of the title
    file_name = f"{short_title}_{short_hash}.txt"
    file_path = os.path.join('./docs', file_name)
    
    # Save the document as a text file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(document)

start_time = time.time()
NUM_DOCS_TO_UPLOAD=10
upload_txt_files("./docs", "http://0.0.0.0:8081/documents", NUM_DOCS_TO_UPLOAD)
print(f"--- {time.time() - start_time} seconds ---")