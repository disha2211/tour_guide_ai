from chromadb.utils.data_loaders import ImageLoader

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

#add images to images_collection

#load documents

#documents to chunks

#add chunks to documents_collection