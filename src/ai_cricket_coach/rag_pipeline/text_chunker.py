import json
from langchain_text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

def chunk_processed_data(input_filepath: str)-> List[Dict[str,Any]]:
    
    
     """
    Split long page texts from a JSONL file into smaller overlapping chunks.

    Steps:
    1. Load each JSON line (page) from the input file into a list.
    2. Initialize a text splitter with:
       - chunk_size = 1000 characters
       - chunk_overlap = 100 characters
    3. For each page:
       - Extract the page text and metadata.
       - Split the text into smaller chunks using the splitter.
       - Create a dictionary for each chunk with its content and metadata.
    4. Return a list of all chunk dictionaries.

    Notes:
    - Keeps the original metadata (like source and page number).
    - Overlapping chunks help preserve context between splits.
    - Useful for preparing data for embeddings or LLM processing.
    """
    
    
    processed_data=[]
    
    
    #loading the jsonl file and append to the preprocessed_data
    with open(input_filepath,'r') as file:
        for line in file:
            
            processed_data.append(json.loads(line))
    
    
    #initialise text splitter
        
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len, #count the number of character
    
    )    
    
    
    #iterates through processed data and extracting the contents
    all_chunks=[]
    for page in processed_data:
        
        # the "" ensures that the program wont crash if the page content is emtpy it stores the empty space("") here.
        page_content=page.get("page_content","")
        
        metadata=page.get("metadata", {}) #{} do same as above
        
        
        #split the page content into chunks
        chunks= text_splitter.split_text(page_content)
        
        
        #for each chunk create a dictionary.
        
        for chunk in chunks:
            chunk_data={
                "chunk_content":chunk,
                "metadata":metadata #original meta data is preserved
            }
            all_chunks.append(chunk_data)
            
    return all_chunks #list of dictionary.