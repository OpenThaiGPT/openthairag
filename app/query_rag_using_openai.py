import openai

# Configure OpenAI client to use vLLM server
openai.api_base = "http://127.0.0.1:5000"
openai.api_key = "dummy"  # vLLM doesn't require a real API key

prompt = "วัดพระแก้ว กทม. เดินทางไปอย่างไร"

# Non-Streaming Response
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
    
# Example usage of non-streaming version
print("Non-streaming response:")
response(prompt)

# Streaming version
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

# Example usage of streaming version
print("Streaming response:")
stream_response(prompt)

