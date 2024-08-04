
###Recursive chunking for the code ####

# from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

# javascript_text = """
# // Example: Sorting an Array

# // Define an array of numbers
# let numbers = [4, 2, 7, 1, 9];

# // Sort the array in ascending order
# numbers.sort((a, b) => a - b);

# // Print the sorted array
# console.log(numbers);
# """

# js_splitter = RecursiveCharacterTextSplitter.from_language(
#     language=Language.JS, chunk_size=65, chunk_overlap=0
# )

# print(js_splitter.create_documents([javascript_text]))





### NTLKTestSplitter ###

# import nltk
# import ssl

# # SSL Workaround
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# # Download punkt
# try:
#     nltk.download('punkt', quiet=True)
# except Exception as e:
#     print(f"Error downloading punkt: {e}")
#     print("Please try manual installation.")

# from langchain.text_splitter import NLTKTextSplitter

# text = "This is an example text. It contains multiple sentences. Let's see how the splitter works."

# try:
#     text_splitter = NLTKTextSplitter()
#     sentences = text_splitter.split_text(text)

#     for sentence in sentences:
#         print(sentence)
# except LookupError:
#     print("NLTK data not found. Please ensure 'punkt' is installed correctly.")
# except Exception as e:
#     print(f"An error occurred: {e}")


from dotenv import load_dotenv
import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Add your OpenAI API Key
if OPENAI_API_KEY == "":
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

breakpoint_threshold_type = "percentile"  # Options: "standard_deviation", "interquartile"

# Create the SemanticChunker with OpenAI Embeddings
text_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(), breakpoint_threshold_type=breakpoint_threshold_type
)

text = """Galaxies are massive collections of stars, dust, and gas. The Milky Way galaxy, which contains our solar system, is estimated to contain hundreds of billions of stars. Galaxies come in a variety of shapes and sizes, including spiral, elliptical, and irregular galaxies.
Black holes are regions of spacetime with such intense gravity that nothing, not even light, can escape. They are formed when a massive star collapses in on itself. The gravity of a black hole is so strong that it can warp the fabric of spacetime around it.
The universe is constantly expanding and evolving. The Big Bang theory is the prevailing cosmological model for the universe. It states that the universe began with a very hot, dense state and has been expanding and cooling ever since. The exact cause of the Big Bang is still unknown."""

documents = text_splitter.create_documents([text])

print(documents)