import fitz
import json
import re #regular expression library

def process_pdf_to_structured_jsonl(pdf_path: str,output_path:str):
      """
    Extract text from each page of a PDF and save it as a structured JSONL file.

    Steps:
    1. Open the PDF using PyMuPDF (`fitz`).
    2. For each page:
       - Extract raw text with `get_text()`.
       - Clean it by replacing extra spaces and newlines.
       - Store the text and metadata (source file, page number) in a dictionary.
    3. Collect all page dictionaries into a list.
       (Each dictionary = one JSON object per page.)
    4. Write the list to a JSONL file, one page per line.

    Notes:
    - Using a list of dictionaries helps keep page data structured.
    - JSONL format is convenient for NLP or search applications.
    - Handle errors gracefully with a try/except block.
    """
    try:
        with fitz.open(pdf_path) as doc:
            processed_data=[]
            #its for reffering the source
            source_filename=pdf_path.split('/')[-1]  #pdf_path.split() makes a list and we took the last element from the list as filename
            for page_num, page in enumerate(doc):
                
                #1)extraction. using the get_text() method on the page object
                raw_text=page.get_text()
                
                #normalization
                #\s means any white space. \s+ means one or more white space. the re.sub function replaces \s+ with ' ' single white space char.
                clean_text=re.sub(r'\s+',' ',raw_text).strip()  #strip() removes white space at the beginning and the end
                
                #creating a dictionary with page content and metadata
                page_data={'page_content':clean_text,
                           'metadata':{'source':source_filename,'page_number':page_num+1}}
                processed_data.append(page_data)#list of dictionaries   , Why we need a list of dictionaries, can i convert directly
        
        #"w" mode opens the file for writing
        with open(output_path,'w') as f:
            for item in processed_data:
                f.write(json.dumps(item)+ '\n') #json.dump() serializes the dictionary into a json string(serialize-converting a data object in memory into a format that can be stored or trasmitted)
    except Exception as e:
        print(f"Error processing{pdf_path}:{e}")
                
            