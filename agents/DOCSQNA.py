from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.graph import START, END, StateGraph
from typing import List, Literal
from typing_extensions import TypedDict
from pymilvus import MilvusClient
from dotenv import load_dotenv
import os
load_dotenv()

class Message(TypedDict):
    role:str
    content:str

class QnaState(TypedDict):
    messages: List[Message]
    matches: List[str]
    query:str
    answer:str 
    improve_count: int  # Track how many times we've improved the query

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GPT_MODEL = os.environ.get('GPT_MODEL')

embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def get_embedding(text):
    return embeddings_model.embed_query(text)

def improve_query(state: QnaState):
    history = state['messages']
    query = state.get('query') or state['messages'][-1]['content']
    llm = ChatOpenAI(model=GPT_MODEL, api_key=OPENAI_API_KEY)
    prompt = f"""
    You are an expert assistant for Dexa Medica. The following is a chat history between a user and the assistant. The user's latest query did not return good results from the knowledge base. Please rewrite or expand the user's query to make it clearer and more likely to retrieve relevant information. Use the chat history for context if needed.

    Chat history:
    {history}

    Original user query:
    {query}

    Improved query:
    """
    improved_query = llm.invoke(prompt).content.strip()
    # Increment improve_count, or set to 1 if not present
    improve_count = state.get('improve_count', 0) + 1
    return {"query": improved_query, "improve_count": improve_count}

def result_judge(state: QnaState) -> Literal["respond", "improve"]:
    history = state['messages']
    content = history[-1]['content']
    matches = state['matches']
    llm = ChatOpenAI(model=GPT_MODEL, api_key=OPENAI_API_KEY)
    prompt = f"""
    You are an expert assistant for Dexa Medica. Given the user's query and the retrieved Q&A matches from the knowledge base, judge if any of the matches are relevant and sufficiently answer the user's query. Be strict: only return True if at least one match directly and clearly addresses the user's question. Otherwise, return False.

    Chat History:
    {history}

    User query:
    {content}

    Retrieved matches:
    {matches}

    Does at least one match sufficiently answer the user's query? Answer with True or False only.
    """
    result = llm.invoke(prompt).content.strip()
    is_relevant = result.lower().startswith("true")
    if is_relevant:
        return "respond"
    else:
        improve_count = state.get('improve_count', 0)
        if improve_count >= 3:
            return "respond"
        else:
            return "improve"

def retrieve_answer(state:QnaState):
    query = state.get('query') or state['messages'][-1]['content']
    query_embedding = get_embedding(query)

    client = MilvusClient(
        uri = 'http://192.168.76.22:19531',
        db_name = 'digicamp_ai_miniproject',
    )
    results = client.search(
        collection_name='Allen_IU_MiniProject',
        data=[query_embedding],
        anns_field='vector',
        limit=1,
        include=['question', 'answer'],
        output_fields=['question', 'answer'],
    )

    matches = [f"Question: {item['entity']['question']}, Answer: {item['entity']['answer']}" for result in results for item in result]

    improve_count = state.get('improve_count', 0)
    return {"matches":matches, "improve_count": improve_count}

def generate_response(state:QnaState):
    history = state['messages']
    query = history[-1]['content']
    llm = ChatOpenAI(model=GPT_MODEL, api_key=OPENAI_API_KEY)
    prompt = f"""
    You are a knowledgeable assistant specializing in Dexa Medica. Given the user's query and the full chat history, carefully analyze the context and select the most relevant Q&A pairs from the knowledge base that best address the user's needs. Write it in a user friendly way, copy paste the answer if necessary as long as it's written in a format easy to understand by user. ONLY USE THE GIVEN RAG ANSWER VALUE IN RAG RESULT FOR YOUR RESPONSE AND NEVER ADD ANY INFORMATION THAT IS NOT IN IT

    Chat history:
    {history}

    Query:
    {query}

    RAG Result:
    {state['matches']}

    Answer:
    """
    return {"answer": llm.invoke(prompt).content, "improve_count": state.get('improve_count', 0)}

workflow = StateGraph(QnaState)

#Define functions
workflow.add_node('retrieve', retrieve_answer)
workflow.add_node('improve', improve_query)
workflow.add_node('respond', generate_response)

#Define edge/route
workflow.add_edge(START, 'retrieve')
workflow.add_conditional_edges(
    'retrieve',
    result_judge,{
        "respond":"respond",
        "improve":"improve"
    }
)
workflow.add_edge('improve', 'retrieve')
workflow.add_edge('respond',END)

graph = workflow.compile()