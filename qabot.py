# Import necessary libraries
from langchain_community.llms import CTransformers
# `CTransformers` is a library for loading and running language models locally using efficient transformer implementations.
from langchain.chains import RetrievalQA
# `RetrievalQA` is a type of chain used for Question-Answering tasks with a retrieval step to find relevant context.
from langchain.prompts import PromptTemplate
# `PromptTemplate` helps in creating structured prompts for interacting with language models.
from langchain_community.embeddings import GPT4AllEmbeddings
# `GPT4AllEmbeddings` generates embeddings for text using local models, supporting inference on CPU or GPU.
from langchain_community.vectorstores import FAISS
# `FAISS` is a vector database used for efficient similarity search and querying of text embeddings.

# Configuration paths
model_file = "models/vinallama-7b-chat_q5_0.gguf"  # Path to the local LLM model file.
vector_db_path = "vectorstores/db_faiss"  # Path to the stored FAISS vector database.

# Function 1: Load the Local Language Model
def load_llm(model_file):
    # Load the language model using CTransformers.
    llm = CTransformers(
        model=model_file,  # Path to the model file (in `.gguf` format).
        model_type="llama",  # Specify the model type (e.g., "llama").
        max_new_tokens=1024,  # Maximum number of tokens to generate in the response.
        temperature=0.01  # Temperature controls randomness; lower values make output more deterministic.
    )
    return llm

# Function 2: Create a Prompt Template
def create_prompt(template):
    # Initialize a PromptTemplate with the given template and input variables.
    prompt = PromptTemplate(
        template=template,  # Template string with placeholders for input variables.
        input_variables=["context", "question"]  # Variables to be filled in by the user input.
    )
    return prompt

# Function 3: Create a Retrieval-based Q&A Chain
def create_qa_chain(prompt, llm, db):
    # Create a RetrievalQA chain using the language model, prompt template, and vector database.
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,  # Loaded LLM instance.
        chain_type="stuff",  # Chain type "stuff" combines retrieved documents into a single context.
        retriever=db.as_retriever(
            search_kwargs={"k": 3},  # Retrieve top 3 most relevant documents from the vector store.
            max_tokens_limit=1024  # Limit the context size to 1024 tokens.
        ),
        return_source_documents=False,  # Do not return source documents in the response.
        chain_type_kwargs={'prompt': prompt}  # Provide the prompt template for generating responses.
    )
    return llm_chain

# Function 4: Load the Vector Database
def read_vectors_db():
    # Load the embedding model used for generating embeddings from text.
    embedding_model = GPT4AllEmbeddings(
        model_file="models/all-MiniLM-L6-v2-f16.gguf"  # Path to the embedding model file.
    )

    # Load the FAISS vector store from the local path.
    # `allow_dangerous_deserialization=True` allows loading the pickle file safely if the source is trusted.
    db = FAISS.load_local(
        vector_db_path,  # Path to the saved vector database.
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db

# Main Code Execution
# Load the vector database.
db = read_vectors_db()

# Load the language model.
llm = load_llm(model_file)

# Define the prompt template for the Q&A task.
template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = create_prompt(template)

# Create the RetrievalQA chain using the loaded LLM, prompt, and vector database.
llm_chain = create_qa_chain(prompt, llm, db)

# Test the Q&A Chain
# Define a question for the language model to answer.
question = "FPT corporation (FPT) là gì?"

# Invoke the chain with the input question and get the response.
response = llm_chain.invoke({"query": question})

# Print the response from the language model.
print(response)
