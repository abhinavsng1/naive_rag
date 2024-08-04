from dotenv import load_dotenv
import os
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Add your OpenAI API Key
if OPENAI_API_KEY is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

docs = [
    Document(
        page_content="A complex, layered narrative exploring themes of identity and belonging",
        metadata={"title":"The Namesake", "author": "Jhumpa Lahiri", "year": 2003, "genre": "Fiction", "rating": 4.5, "language":"English", "country":"USA"},
    ),
    Document(
        page_content="A luxurious, heartfelt novel with themes of love and loss set against a historical backdrop",
        metadata={"title":"The Nightingale", "author": "Kristin Hannah", "year": 2015, "genre": "Historical Fiction", "rating": 4.8, "language":"English", "country":"France"},
    ),
    Document(
        page_content="A full-bodied epic with rich characters and a sprawling plot",
        metadata={"title":"War and Peace", "author": "Leo Tolstoy", "year": 1869, "genre": "Historical Fiction", "rating": 4.7, "language":"Russian", "country":"Russia"},
    ),
    Document(
        page_content="An elegant, balanced narrative with intricate character development and subtle themes",
        metadata={"title":"Pride and Prejudice", "author": "Jane Austen", "year": 1813, "genre": "Romance", "rating": 4.6, "language":"English", "country":"UK"},
    ),
    Document(
        page_content="A highly regarded novel with deep themes and a nuanced exploration of human nature",
        metadata={"title":"To Kill a Mockingbird", "author": "Harper Lee", "year": 1960, "genre": "Fiction", "rating": 4.9, "language":"English", "country":"USA"},
    ),
    Document(
        page_content="A crisp, engaging story with vibrant characters and a compelling plot",
        metadata={"title":"The Alchemist", "author": "Paulo Coelho", "year": 1988, "genre": "Adventure", "rating": 4.4, "language":"Portuguese", "country":"Brazil"},
    ),
    Document(
        page_content="A rich, complex narrative set in a dystopian future with strong thematic elements",
        metadata={"title":"1984", "author": "George Orwell", "year": 1949, "genre": "Dystopian", "rating": 4.7, "language":"English", "country":"UK"},
    ),
    Document(
        page_content="An intense, gripping story with dark themes and intricate plot twists",
        metadata={"title":"Gone Girl", "author": "Gillian Flynn", "year": 2012, "genre": "Thriller", "rating": 4.3, "language":"English", "country":"USA"},
    ),
    Document(
        page_content="An exotic, enchanting tale with rich descriptions and an intricate plot",
        metadata={"title":"One Hundred Years of Solitude", "author": "Gabriel García Márquez", "year": 1967, "genre": "Magical Realism", "rating": 4.8, "language":"Spanish", "country":"Colombia"},
    ),
]

embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(docs,embeddings)

metadat_field_info =[
    AttributeInfo(name ="title",description="Description of the book",type="string or list[string]"),
    AttributeInfo(name ="author",description="Author of the book",type="string or list[string]"),
    AttributeInfo(name ="year",description="The year book was published",type="integer"),
    AttributeInfo(name ="genre",description="The genre of book",type="string or list[string]"),
    AttributeInfo(name ="rating",description="The rating of the book(1-5 scale)",type="float"),
    AttributeInfo(name ="language",description="The language of the book is written in",type="string"),
    AttributeInfo(name ="country",description="The country the author is from",type="string or list[string]"),
]

document_content_description = 'brief description of the book'

llm = OpenAI(temperature = 0)

retiever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadat_field_info,
    verbose = True
)

# query
# ans = retiever.invoke('What are some highly rated historical fiction books')
# print(ans)
ans = retiever.invoke('What are some books above rating 4.8')
print(ans)

ans = retiever.invoke('What are two books that comes from USA')
print(ans)