from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

'''
- step1 --> Load raw PDFs
- step2 --> Create Chunks
- step3 --> Create Vector embedding
- step4 --> Store embedding in FAISS
'''

DATA_PATH = "Data/"

# Step 1 --> Load raw PDFs
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

try:
    documents = load_pdf_files(DATA_PATH)
except Exception as e:
    print("Error loading PDFs:", e)

print("Length of pages:", len(documents))

# Step 2 --> Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(documents)
print("Length of tex chuncks:", len(text_chunks))


# - step3 --> Create Vector embedding

# get the embeding model from the hugging-face
'''
This is a sentence-transformers model: It maps sentences 
& paragraphs to a 384 dimensional dense vector space and 
can be used for tasks like clustering or semantic search.
'''

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embedding_model

embedding_model = get_embedding_model()


# - step4 --> Store embedding in FAISS

DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)

