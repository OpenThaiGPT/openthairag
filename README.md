# OpenThaiRAG
![logo](https://github.com/user-attachments/assets/901b1532-2d24-4955-9659-789ae077bb30)

OpenThaiRAG is an open-source Retrieval-Augmented Generation (RAG) framework designed specifically for Thai language processing. This project combines the power of vector databases, large language models, and information retrieval techniques to provide accurate and context-aware responses to user queries in Thai using OpenThaiGPT 1.5 as LLM. For more about OpenThaiGPT project: https://openthaigpt.aieat.or.th


## OpenThaiRAG-WEBUI
You can use OpenThaiRAG-WEBUI for GUI version of OpenThaiRAG developed by K. Phanuwat Wiriyasiriwatthana. 
https://github.com/sunnypdater/openthairag-webui.git


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

## Indexing New Documents into RAG
To insert new documents into the RAG system, you can use the `index_docs.py` script provided in the `app` directory. This script reads text files from the `/docs` folder and indexes their contents via the API. Here's how to use it:

1. Prepare your documents:
   - Create text files (.txt) containing the content you want to index.
   - Place these files in the `/docs` directory of your project.

2. Run the indexing script:
   ```
   python app/index_docs.py
   ```

   This script will:
   - Read all .txt files in the `/docs` directory.
   - Split each document into chunks of maximum 200 characters, including the title in each chunk.
   - Send each chunk to the indexing endpoint (http://localhost:5000/index by default).

3. Monitor the indexing process:
   - The script will log information about each indexed file.
   - At the end, it will report the total number of successfully indexed files and any files that couldn't be indexed.

You can also customize the indexing process by modifying the `index_docs.py` script. For example, you can change the chunk size, adjust the indexing endpoint URL, or add additional preprocessing steps.

Note: Ensure that your OpenThaiRAG API is running and accessible at the specified URL before running the indexing script.

For more granular control or to index documents programmatically, you can use the `/index` endpoint directly:

## Example Document TXT Files
```txt
Title: วัดธาตุทอง (Wat That Thong)
Content: วัดธาตุทอง พระอารามหลวง ตั้งเมื่อปีพุทธศักราช ๒๔๘๑ และได้รับพระราชทานวิสุงคามสีมา เมื่อวันที่ ๒๔ ตุลาคม พุทธศักราช ๒๔๘๓(เขตวิสุงคามสีมา กว้าง ๔๐ เมตร ยาว ๘๐ เมตร) ผูกพัทธสีมา ฝังลูกนิมิตอุโบสถ เมื่อวันที่ ๒ ๘ กุมภาพันธ์ พุทธศักราช ๒๕๐๕ มีเนื้อที่ ๕๔ ไร่ ๓ งาน ๘๒ ตาราง(เลขที่ ๑๔๙ โฉนดที่ ๔๐๓๗)

ทิศเหนือ ติดกับที่ดินและบ้านเรือนประชาชน(ซอยชัยพฤกษ์)

ทิศใต้ ติดกับถนนสุขุมวิท

ทิศตะวันออก ติดกับที่ดินและบ้านเรือนประชาชน(ซอยเอกมัย)

วัดธาตุทองฯ แท้จริงแล้วมีประวัติความเป็นมายาวนาน ย้อนกลับไปถึงยุคสมัยสุโขทัยเป็นราชธานี ก่อนจะมาตั้งอยู่บนนถนนสุขุมวิทในปัจจุบัน

Nearby Location: ตั้งอยู่ริมถนนสุขุมวิท แขวงพระโขนงเหนือ เขตวัฒนา
Address: 1325
Region: ภาคกลาง
Alley: 
Road: สุขุมวิท
Subdistrict: 
District: วัฒนา
Province: กรุงเทพมหานคร
Category: แหล่งท่องเที่ยวทางประวัติศาสตร์ และวัฒนธรรม
Sub Type: ศาสนสถาน (วัด/โบสถ์/มัสยิด ฯลฯ)
Facilities Contact: 
Telephone: 0 2390 0261, 0 2391 1007
Email: 
Website: 
Facebook: 
Instagram: 
Line: 
TikTok: 
YouTube: 
Start-End: 05.30 21.00น.
Activity: 
Suitable Duration: 
Fee (TH): 
Fee (TH Kid): 
Fee (EN): 
Fee (EN Kid): 
Remark: 
Location: 13.7194087, 100.5857861
UUID: 1ed676ed-4161-40f6-9e3d-12f4db53851d
Created Date: 2024-09-23
Updated Date: 2024-09-23
URL: 
Published Date: 
```
You can see more examples at `/docs`.

## Getting RAG's Response.
To get a response from the RAG system, you can use the `/completions` endpoint. This endpoint accepts a POST request with a JSON payload containing the user's query and optional parameters. 

Here's a list of query parameters supported by the `/completions` endpoint:

1. `prompt` (required): The input text to generate completions for.
2. `max_tokens` (optional): The maximum number of tokens to generate. Defaults to 16.
3. `temperature` (optional): Controls randomness in generation. Higher values (e.g., 0.8) make output more random, lower values (e.g., 0.2) make it more focused. Defaults to 1.0.
4. `top_p` (optional): An alternative to temperature, called nucleus sampling. Keeps the model from considering unlikely options. Defaults to 1.0.
5. `n` (optional): How many completions to generate for each prompt. Defaults to 1.
6. `stream` (optional): Whether to stream back partial progress. Defaults to false.
7. `logprobs` (optional): Include the log probabilities on the `logprobs` most likely tokens. Defaults to null.
8. `echo` (optional): Echo back the prompt in addition to the completion. Defaults to false.
9. `stop` (optional): Up to 4 sequences where the API will stop generating further tokens.
10. `presence_penalty` (optional): Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far. Defaults to 0.
11. `frequency_penalty` (optional): Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far. Defaults to 0.
12. `best_of` (optional): Generates best_of completions server-side and returns the "best" (the one with the highest log probability per token). Defaults to 1.
13. `logit_bias` (optional): Modify the likelihood of specified tokens appearing in the completion.
14. `user` (optional): A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.

Note: Some parameters may not be applicable depending on the specific model and configuration of your OpenThaiRAG setup.


### via API: Non-Streaming
```bash
>>>Request
curl --location 'http://localhost:5000/completions' \
--header 'Content-Type: application/json' \
--data '{
    "prompt": "วัดพระแก้ว กทม. คืออะไร",
    "max_tokens": 2048,
    "temperature": 0.7
}'

<<<Response
{
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "logprobs": null,
            "prompt_logprobs": null,
            "stop_reason": null,
            "text": "วัดพระแก้ว (Wat Phra Kaeo) ตั้งอยู่ในจังหวัดชัยนาท สร้างในสมัยเดียวกับวัดมหาธาตุ ตั้งแต่ปี พ.ศ. 1900 วัดพระแก้วมีเจดีย์ทรงสูง ลักษณะเป็นเจดีย์แบบละโว้ผสมกับเจดีย์ทวารวดีตอนปลาย สร้างแบบสอปูน เป็นเจดีย์ฐานสี่เหลี่ยม มีพระพุทธรูปปั้นแบบนูนสูงประดับทั้งสี่ด้าน วัดพระแก้วมีพระสถูป เจดีย์ และพระพุทธรูปศิลาแลงสีแดง คือ หลวงพ่อทันใจ ที่อยู่ในวิหารด้านหน้าพระเจดีย์สี่เหลี่ยม วัดพระแก้วตั้งอยู่นอกเมืองทางด้านทิศใต้ ห่างจากวัดมหาธาตุประมาณ 3 กม. ปัจจุบันวัดพระแก้วอยู่กลางทุ่งนา มีพระเจดีย์เหลี่ยมเป็นหลักของวัด วัดพระแก้วเป็นโบราณสถานที่มีความสำคัญทางประวัติศาสตร์และศิลปะ ซึ่งได้รับการขึ้นทะเบียนเป็นโบราณสถานโดยกรมศิลปากรเมื่อวันที่ 8 มีนาคม 2478."
        }
    ],
    "created": 1728035246,
    "id": "cmpl-e0e5752f01e34d2bb701f86fad3b4954",
    "model": ".",
    "object": "text_completion",
    "usage": {
        "completion_tokens": 386,
        "prompt_tokens": 4946,
        "total_tokens": 5332
    }
}
```

### via API: Streaming
```bash
>>>Request
curl --location 'http://localhost:5000/completions' \
--header 'Content-Type: application/json' \
--data '{
    "prompt": "วัดพระแก้ว กทม. คืออะไร",
    "max_tokens": 2048,
    "temperature": 0.7,
    "stream": true
}'

<<<Response
data: {"id":"cmpl-8dbd8bdfbcfb4310bf611cd6f6f7c2e4","object":"text_completion","created":1728035332,"model":".","choices":[{"index":0,"text":"","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}

...

data: {"id":"cmpl-8dbd8bdfbcfb4310bf611cd6f6f7c2e4","object":"text_completion","created":1728035332,"model":".","choices":[{"index":0,"text":"ื","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}

data: {"id":"cmpl-8dbd8bdfbcfb4310bf611cd6f6f7c2e4","object":"text_completion","created":1728035332,"model":".","choices":[{"index":0,"text":"องชัยนาท.","logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":null}

data: [DONE]
```

### via OpenAI Library
You can take a look at ``/app/query_rag_using_openai.py``.
To use the OpenAI library to get RAG responses, you can follow these steps:

1. Install the OpenAI library:
   ```
   pip install openai==0.28
   ```

2. Configure the OpenAI client to use the vLLM server:
   ```python
   import openai

   openai.api_base = "http://127.0.0.1:5000"
   openai.api_key = "dummy"  # vLLM doesn't require a real API key
   ```

3. Define your prompt:
   ```python
   prompt = "วัดพระแก้ว กทม. เดินทางไปอย่างไร"
   ```

4. For a non-streaming response:
   ```python
   def response(prompt):
       try:
           response = openai.Completion.create(
               model=".",  # Specify the model you're using with vLLM
               prompt=prompt,
               max_tokens=512,
               temperature=0.7,
               top_p=0.8,
               top_k=40,
               stop=["<|im_end|>"]
           )
           print("Generated Text:", response.choices[0].text)
       except Exception as e:
           print("Error:", str(e))

   # Example usage
   print("Non-streaming response:")
   response(prompt)
   ```

5. For a streaming response:
   ```python
   def stream_response(prompt):
       try:
           response = openai.Completion.create(
               model=".",  # Specify the model you're using with vLLM
               prompt=prompt,
               max_tokens=512,
               temperature=0.7,
               top_p=0.8,
               top_k=40,
               stop=["<|im_end|>"],
               stream=True  # Enable streaming
           )
           
           for chunk in response:
               if chunk.choices[0].text:
                   print(chunk.choices[0].text, end='', flush=True)
           print()  # Print a newline at the end
       except Exception as e:
           print("Error:", str(e))

   # Example usage
   print("Streaming response:")
   stream_response(prompt)
   ```

You can find the complete example in the `/app/query_rag_using_openai.py` file.

## Full API Documentation

For detailed API documentation and examples, please refer to our Postman collection: https://documenter.getpostman.com/view/5145656/2sAYBd67fw

## Maintainer

**OpenThaiGPT Team**
* Kobkrit Viriyayudhakorn (kobkrit@aieat.or.th)
* Sumeth Yuenyong (sumeth.yue@mahidol.edu)
* Apivadee Piyatumrong (apivadee.piy@nectec.or.th)
* Jillaphat Jaroenkantasima (autsadang41@gmail.com)
  
## License

Apache 2.0
