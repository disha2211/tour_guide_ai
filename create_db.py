import os

from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from utils import ImageEmbeddings,DATA_PATH,DB_PATH

import chromadb


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
	    embedding_function=OpenAIEmbeddingFunction(
	        api_key=os.getenv("OPENAI_API_KEY"),
	        model_name="text-embedding-3-small"
	    )
	)


#load documents

#documents to chunks

#add chunks to documents_collection