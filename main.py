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

loader = PyPDFLoader(DOC_PATH)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

CHROMA_PATH = "your_chroma_path"
db_chroma = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)


query = 'What is Transformer?'

docs_chroma = db_chroma.similarity_search_with_score(query, k=5)

context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])


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