# LangChain-QA-Prompt-Query-From-Text-and-PDF-Documents
The Repository reference from the https://github.com/thangnch/MiAI_Langchain_RAG, I want to learn the LangChain framework. The Repo is my first project to understand how load models, create a Vector Database, Prompt components, Format files interact with LLMs, and RAG (Retrieval Augmented Generation).

# 1. Install Required Dependencies
git clone https://github.com/NhanAyai/LangChain-QA-Prompt-Query-From-Text-and-PDF-Documents.git
cd LangChain-QA-Prompt-Query-From-Text-and-PDF-Documents
pip install -r setup.txt

# 2. Prepare Directories and Download Models
Make sure you have the following directories created:
data: This directory will store all your PDF documents.
models: This directory will store the model files used for querying.
Download the pre-trained models (one for document query and one for the Vietnamese model) and place them in the models directory:
Download [all-MiniLM-L6-v2-f16.gguf](https://huggingface.co/caliex/all-MiniLM-L6-v2-f16.gguf/resolve/main/all-MiniLM-L6-v2-f16.gguf?download=true) model
Download [vinallama-7b-chat-GGUF](https://huggingface.co/vilm/vinallama-7b-chat-GGUF/resolve/main/vinallama-7b-chat_q5_0.gguf?download=true) Vietnamese model

# 3. Prepare Vector Database
python prepare_vector_db.py
Run the script prepare_vector_db.py to create the vector database. The script will store vectors in a database that can be queried later.
You can generate vectors either from text files or directly from PDF documents. The script has an optional function that allows you to use text data to populate the database.

# 4. Test the Sample Query
python simplechain.py
Run simplechain.py to test the query functionality using a sample question. You can input your question in the question variable, and the script will return the most relevant answer from the vector database.

# 5. Create a Chatbot Q&A System
python qabot.py
To run a full Q&A chatbot, execute the qabot.py script. This will create a ChatBox that can:
Query from normal text.
Retrieve context from PDF documents stored in the data directory.
Feel free to add your own PDF files into the data folder and ask questions based on the content of those files.

# 6. Acknowledgements
Thank you for your work in setting up this project! If you find this repository helpful, please give it a star ⭐.

Good luck with your LangChain journey!
