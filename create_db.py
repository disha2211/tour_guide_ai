"""__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
"""

import os

from chromadb.utils.data_loaders import ImageLoader
#-----------using google gemini api key
from chromadb.utils.embedding_functions import GoogleGeminiEmbeddingFunction

from utils import ImageEmbeddings,DATA_PATH,DB_PATH

import chromadb

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from dotenv import load_dotenv
load_dotenv()

# new chromadb client

#-------------------as working on local machine , just need a persistent client
#------------------check the docs under chroma clients (persistent client)
client_db = chromadb.PersistentClient(path=DB_PATH)


#create images_collection
#---------------docs -> manage collections
img_collection = client_db.create_collection(
	name="imgs",
	embedding_function=ImageEmbeddings(),
	data_loader=ImageLoader()
	)

#for each category
for dir_ in os.listdir(DATA_PATH):
	dir_path = os.path.join(DATA_PATH, dir_)

#add images to images_collection
#----------------docs -> add data
img_collection.add(
	    ids=[f"{dir_}-{img_path}" for img_path in os.listdir(dir_path) if img_path.endswith('.jpg')],
	    uris=[os.path.join(dir_path, img_path) for img_path in os.listdir(dir_path) if img_path.endswith('.jpg')]
	)

#   create documents_collection
#---------------docs -> manage collections -> docs collection
collection = client_db.create_collection(
    name=f"documents_{dir_}",
    embedding_function=GoogleGeminiEmbeddingFunction(
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name="gemini-embedding-001"  # Or "gemini-embedding-2" if you want the latest model
    )
)


#load documents
loader = DirectoryLoader(dir_path, glob="*.txt")
documents = loader.load()

#documents to chunks
#----------important to chunk text for RAG
text_splitter = RecursiveCharacterTextSplitter(
	chunk_size=300,
	chunk_overlap=100,
	length_function=len,
	add_start_index=True,
)

chunks = text_splitter.split_documents(documents)

#add chunks to documents_collection


collection.add(
	ids=[str(j) for j in range(len(chunks))],
	documents=[chunks[j].page_content for j in range(len(chunks))],
	metadatas=[chunks[j].metadata for j in range(len(chunks))],
)