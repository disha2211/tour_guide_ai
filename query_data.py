
import chromadb
import cv2

from utils import DB_PATH, ImageEmbeddings


def classify_img(client_db,query_img):
    #-----------docs->manage collection
    collection = client_db.get_collection(name="imgs")

    embeddingFunction = ImageEmbeddings()
    img = cv2.imread(query_img)
    embeddings = embeddingFunction([img])

    #----------docs-> query and get


#input data: query img, query text
query_img = None # path to image
query_question = None

#create chroma db client
#-----------similar to client created in create_db.py
client_db = chromadb.PersistentClient(path=DB_PATH)

#classify image


#get most similar chunks

#create response