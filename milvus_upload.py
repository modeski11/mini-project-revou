from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient
import pandas as pd
import os
import re

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
embeddings_model = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

def get_embedding(text):
    return embeddings_model.embed_query(text)

def split_by_numbered_items(text, keep_separators=True):
    pattern = r'(\d+\.)'
    
    if keep_separators:
        chunks = re.split(pattern, text)
        result = []
        for i in range(1, len(chunks), 2):
            if i + 1 < len(chunks):
                chunk = chunks[i] + chunks[i + 1]
                result.append(chunk.strip())
            else:
                result.append(chunks[i].strip())
        
        if chunks[0].strip():
            result.insert(0, chunks[0].strip())
            
    else:
        chunks = re.split(pattern, text)
        result = []
        for chunk in chunks:
            stripped = chunk.strip()
            if stripped and not re.match(r'\d+\.$', stripped):
                if "\n \n" in stripped:
                    stripped = stripped.split("\n \n")[0].strip()
                if stripped:
                    result.append(stripped)
    
    return result

loader = PyMuPDFLoader("docs/FAQ Dexa Medica.pdf")
docs = loader.load()
chunk_text = split_by_numbered_items(''.join([doc.page_content for doc in docs]), keep_separators=False)
qna_result = [
    {
        "question": chunk.split('?', 1)[0].strip() + '?',
        "answer": chunk.split('?', 1)[1].strip() if '?' in chunk else ''
    }
    for chunk in chunk_text if '?' in chunk
]

df = pd.DataFrame.from_records(qna_result)
df['vector'] = df.apply(lambda row: get_embedding(f"{row['question']} {row['answer']}"), axis=1)

client = MilvusClient(
    uri = os.environ.get('MILVUS_URI')
)
client.use_database('digicamp_ai_miniproject')

res = client.insert(
    collection_name = 'Allen_IU_MiniProject',
    data=df.to_dict(orient='records')
)