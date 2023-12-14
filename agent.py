import os

import dotenv
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import Chroma

loader = PyPDFDirectoryLoader(f'docs/', glob="**/*.pdf", load_hidden=False)
docs = loader.load_and_split()

print(len(docs))
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
store = LocalFileStore("cache/")
underlying_embeddings = OpenAIEmbeddings(show_progress_bar=True, chunk_size=50)
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)
vectorstore = Chroma.from_documents(docs, cached_embedder)
# vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings(show_progress_bar=True, chunk_size=50))
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6})

tool = create_retriever_tool(
    retriever,
    "starr_bot",
    "Searches and returns documents regarding the state-of-the-union.",
)
tools = [tool]
llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)
agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)