
import chromadb
import cv2

from utils import DB_PATH, ImageEmbeddings


def classify_img(client_db,query_img):
    #-----------docs->manage collection
    collection = client_db.get_collection(name="imgs")

    embeddingFunction = ImageEmbeddings()
    img = cv2.imread(query_img)
    embeddings = embeddingFunction([img])

    results = collection.query(
        query_embeddings=embeddings,
        n_results=1
    )

    #print(results['ids'][0][0].split('-')[0])---------output:"prague_castle"

    return results['ids'][0][0].split('-')[0]

def get_most_similar_chunks(client_db, query_question, img_category):
	collection = client_db.get_collection(name=f"documents_{img_category}")

	results = collection.query(
		query_texts=[query_question],
		n_results=3
	)

    #print(results)
    #return text and meta data

	return results['documents'][0], results['metadatas'][0]




#input data: query img, query text
query_img = None # path to image
query_question = None  # "when was it built?"

#create chroma db client
#-----------similar to client created in create_db.py
client_db = chromadb.PersistentClient(path=DB_PATH)

#classify image
img_category=classify_img(client_db,query_img)

#get most similar chunks
chunks=get_most_similar_chunks(client_db,query_question)

#create response