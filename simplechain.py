# Import necessary libraries
from langchain_community.llms import CTransformers  # CTransformers is used to load and interact with local LLM models like LLaMA.
from langchain.chains import LLMChain  # LLMChain helps create a sequence of operations involving the LLM.
from langchain.prompts import PromptTemplate  # PromptTemplate allows creating custom prompt templates for querying the LLM.

# Configuration: Specify the path to the local LLaMA model file
model_file = "models/vinallama-7b-chat_q5_0.gguf"

# Function to load the LLM model
def load_llm(model_file):
    # Initialize the LLM using CTransformers with specific settings:
    # - `model`: Path to the model file
    # - `model_type`: Type of model (e.g., LLaMA)
    # - `max_new_tokens`: Maximum number of new tokens the model can generate
    # - `temperature`: Controls the randomness of the model's output (lower values = more deterministic)
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01
    )
    return llm

# Function to create a prompt template
def create_prompt(template):
    # `PromptTemplate` is used to create a reusable prompt format.
    # - `template`: The format of the prompt with placeholders for input variables.
    # - `input_variables`: List of variables that can be passed into the template (e.g., "question").
    prompt = PromptTemplate(template=template, input_variables=["question"])
    return prompt

# Function to create a simple LLM chain
def create_simple_chain(prompt, llm):
    # `LLMChain` connects a prompt template with the LLM model, forming a chain of operations.
    # - `prompt`: The prompt template created earlier.
    # - `llm`: The loaded LLM model.
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain

# Test the LLM chain

# Define a template for the prompt
template = """<|im_start|>system
You are a helpful AI assistant. Please answer the user's question accurately.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

# Create a prompt template
prompt = create_prompt(template)

# Load the LLM model
llm = load_llm(model_file)

# Create the LLM chain using the prompt and LLM model
llm_chain = create_simple_chain(prompt, llm)

# Define a test question to ask the model
question = "Một cộng 1 bằng mấy?"

# Run the chain with the test question
response = llm_chain.invoke({"question": question})

# Print the response from the model
print(response)
