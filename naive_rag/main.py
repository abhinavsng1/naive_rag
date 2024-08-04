from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Add your OpenAI API Key
if OPENAI_API_KEY is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")


DOC_PATH = "transformer.pdf"

# load the pdf 
loader = PyPDFLoader(DOC_PATH)
pages = loader.load()

#convert it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)

#create an embedding
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

#create an chroma database
CHROMA_PATH = "your_chroma_path"
db_chroma = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)

#enter the query
query = 'What is Transformer?'

#search the db and mention the k,where k mentions the top k matched result
docs_chroma = db_chroma.similarity_search_with_score(query, k=5)

#context texts  
context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

#prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}.
Provide a detailed answer.
Don’t justify your answers.
Don’t give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar.
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query)

model = ChatOpenAI()

response_text = model.invoke(prompt)
print(response_text)