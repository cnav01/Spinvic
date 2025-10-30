import json
import chromaDB #on disc databse to store our vectors
from sentencetransformer import SentenceTransformer #
from typing import List , Dict, Any

def load_chunked_data(input_filepath:str)-> List[Dict[str,Any]]:
    """
    Loads the chunked data from the .jsonl file.
    
    Args:
        input_filepath (str): Path to the .jsonl file.

    Returns:
        List[Dict[str, Any]]: A list of chunk dictionaries.
    """
    
    
    
    chunked_documents=[]
    
    try:
        with open(input_filepath,'r') as f:
            for line in f:
                chunked_documents.append(json.load(line))
        return chunked_documents
    
    except FileFoundError:
        print(f"the file{input_filepath}not found")#for file found error
        return []#return an empty list
    
    except Exception as e:
        print(f"An error while loading{input_filepath}:{e}")
        return []


def embed_and_store_chunks(chunked_documents:List[Dict[str,Any]],
                           model_name:str,
                           collection_name:str,
                           db_path:str):
    
    #1 loading the embedding model
    print(f"loading the embedding mode:{model_name}")
    model=SentenceTrasformer(model_name)
    
    #2 initialize a persistent database client
    
    # It creates an object, client, that you will use to interact with the database(like add, query)
    client=chromaDB.PersistentClient(path=db_path)
    
    #get or create the collection(like a table in sql)
    collection=client.get_or_create_collection(name=collection_name)
    
    # we are going to prepare the stuff to put it in the collections
    
    #we have, the chunk, metadata, id put them into organised piles
    #batch processing
    '''
    The opposite (and slower) way would be to:

Get one chunk.
Embed that one chunk.
Add that one chunk to the database.
get the next chunk.

...and so on.

Our professional method is much faster:
Build the batches (this is the next step).
Embed all 1000 chunks at once (this is the step after that).
Add all 1000 chunks to the database at once.
So, the next logical step is to loop through our chunked_documents and fill the doc_batch, meta_batch, and id_batch lists.
'''
    doc_batch=[]
    meta_batch=[]
    id_batch=[]
    
    
    print("preparing data batches")
    for i,docs in enumerate(chunked_documents): 
        
        content=docs.get("chunk content")
        metadata=docs.get("metadata",{})# the curly bracket is for it to never return a none
        
        chunk_id = f"{metadata.get('source', 'unknown')}_page_{metadata.get('page_number', '0')}_chunk_{i}"
        
        doc_batch.append(content)
        meta_batch.append(metadata)
        id_batch.append(chunk_id)
        
    #perform batch embedding(much faster than loops)
    
    print(f"embedding{len(doc_batch)} chunks...")
    #currently embedding is a numpy array
    embeddings=model.encode(doc_batch)#only we search the contents of the doc_batch
    
    #performing the batch storage
    
    print("storing embedded data in vector db at {db_path}")
    collection.add(
        embeddings=embeddings.tolist(),#converting numpy array to a list
        documents=doc_batch,
        metadata=meta_batch,
        ids=id_batch
            
    )
    
    print(f"--- Embedding and Storage Complete ---")
    print(f"Total chunks processed: {len(doc_batch)}")
        
        
        
        
        
    
    
    
    
    
    
    
    

