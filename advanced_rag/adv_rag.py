from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
import uuid
from langchain_core.documents import Document


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Add your OpenAI API Key
if OPENAI_API_KEY is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

loaders = [
    TextLoader("blog.langchain.dev_announcing-langsmith_.txt"),
    TextLoader("blog.langchain.dev_automating-web-research_.txt"),
]

docs=[]

for loader in loaders:
    docs.extend(loader.load())
    
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(docs)


chain = (
    {"docs":lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarise the following documents:\n\n{docs}")
    |ChatOpenAI(model='gpt-3.5-turbo',max_retries=0)
    |StrOutputParser()
)

summaries = chain.batch(docs,{"max_concurrency":3})


vectorstore= Chroma(collection_name="summaries",embedding_function=OpenAIEmbeddings())

store = InMemoryByteStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key
)


doc_ids = [str(uuid.uuid4()) for _ in docs]

summary_docs = [
    Document(page_content=s,metadata={id_key:doc_ids[i]})
    for i,s in enumerate(summaries)
]


retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids,docs)))

query = "what is langsmith?"


sub_docs = vectorstore.similarity_search(query)
print("sub_docs", sub_docs)
sub_docs[0]

retrieved_docs = retriever.invoke(query)

print("retrieved_docs",retrieved_docs)

retrieved_docs[0].page_content[0:500]

len(retrieved_docs[0].page_content)



