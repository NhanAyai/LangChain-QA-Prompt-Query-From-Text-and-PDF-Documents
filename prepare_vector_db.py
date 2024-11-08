# Import necessary libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
# `RecursiveCharacterTextSplitter` and `CharacterTextSplitter` are used to split text into smaller chunks.
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# `PyPDFLoader` is used for loading PDF files, while `DirectoryLoader` loads all files from a specified directory.
from langchain_community.vectorstores import FAISS
# `FAISS` is a vector store used for storing and querying text embeddings.
from langchain_community.embeddings import GPT4AllEmbeddings
# `GPT4AllEmbeddings` generates embeddings locally using a model like MiniLM, loaded from a `.gguf` file.

# Path configuration
pdf_data_path = "data"  # Folder where PDF files are stored.
vector_db_path = "vectorstores/db_faiss"  # Path to store the FAISS vector database.

# Function 1: Create a Vector Database from Raw Text**
def create_vector_db():
    # Define raw text for testing purposes (e.g., text from a PDF document).
    raw_text = '''Trong suốt hành trình 34 năm, FPT đã không ngừng nghiên 
                  cứu, phát triển, sáng tạo, mang lại những giải pháp thiết 
                  thực và hiệu quả cho nhiều triệu người dân Việt Nam, 
                  hàng chục nghìn doanh nghiệp và tổ chức trên phạm vi 
                  toàn cầu, từ đó đảm bảo phát triển bền vững như cam kết 
                  với cổ đông.
                  Năm 2023, năm thứ 35 trong hành trình phát triển, FPT 
                  khát vọng tiếp tục song hành cùng mỗi cá nhân, mỗi tổ 
                  chức, mỗi doanh nghiệp trên hành trình không ngừng 
                  kiến tạo trải nghiệm hạnh phúc thông qua những dịch vụ, 
                  giải pháp sáng tạo, vượt trội về công nghệ.'''

    # Split the text into smaller chunks for efficient processing.
    text_splitter = CharacterTextSplitter(
        separator='\n',  # Split the text based on newline characters.
        chunk_size=512,  # Maximum size of each chunk (512 characters).
        chunk_overlap=50,  # Overlapping characters between chunks for better context.
        length_function=len  # Function to determine the length of each chunk.
    )

    # Split the raw text into chunks.
    chunks = text_splitter.split_text(raw_text)

    # Generate embeddings for each chunk using GPT4AllEmbeddings.
    embedding_model = GPT4AllEmbeddings(
        model_file='models/all-MiniLM-L6-v2-f16.gguf',  # Path to the embedding model file.
        device='cpu'  # Specify device for inference (e.g., CPU).
    )

    # Create a FAISS vector store from the text chunks using the generated embeddings.
    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)

    # Save the vector store locally for future use.
    db.save_local(vector_db_path)
    return db

# Function 2: Create a Vector Database from PDF Files
def create_db_from_files():
    # Load all PDF documents from the specified directory using DirectoryLoader and PyPDFLoader.
    loader = DirectoryLoader(
        pdf_data_path,  # Directory containing PDF files.
        glob='*.pdf',  # Load only files with a `.pdf` extension.
        loader_cls=PyPDFLoader  # Use PyPDFLoader for reading PDF content.
    )

    # Load all documents from the directory into a list.
    documents = loader.load()

    # Split the documents into smaller chunks using RecursiveCharacterTextSplitter.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  # Maximum size of each chunk (512 characters).
        chunk_overlap=50  # Overlapping characters between chunks for better context.
    )

    # Split the loaded documents into chunks.
    chunks = text_splitter.split_documents(documents)

    # Generate embeddings for each chunk using GPT4AllEmbeddings.
    embedding_model = GPT4AllEmbeddings(
        model_file='./models/all-MiniLM-L6-v2-f16.gguf',  # Path to the embedding model file.
        device='cpu'  # Specify device for inference (e.g., CPU).
    )

    # Create a FAISS vector store from the document chunks using the generated embeddings.
    db = FAISS.from_documents(chunks, embedding_model)

    # Save the vector store locally for future use.
    db.save_local(vector_db_path)
    
    return db

# Execute the function to create the vector database from PDF files.
create_db_from_files()
