from phi.agent import Agent 
from phi.model.ollama import Ollama
from phi.embedder.ollama import OllamaEmbedder
from phi.knowledge.csv import CSVKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
from phi.document.chunking.fixed import FixedSizeChunking
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k


knowledge_base = CSVKnowledgeBase(
    path="data/IMDB-Movie-Data.csv",
    vector_db=LanceDb(
        table_name="IMDB-Movie-Data",
        uri="tmp/lancedb/imdb",
        search_type=SearchType.vector,
        embedder=OllamaEmbedder()
    ),
    chunking_strategy=FixedSizeChunking(chunk_size=1024, overlap=10)
)

# knowledge_base.load()

agent_rag_imdb = Agent(
    model=Ollama(id="llama3.2"),
    tools = [DuckDuckGo(), Newspaper4k()],
    knowledge=knowledge_base,
    add_context=True,
    show_tool_calls=True,
    markdown=True,
    instructions="Please display the data in tabular format only"
)

agent_rag_imdb.print_response("can you suggest top 2 science fiction movies with high ratings from knowledge base", stream=True)
agent_rag_imdb.print_response("What is recent news in USA", stream=True)
agent_rag_imdb.print_response("can summarize the article https://edition.cnn.com/us/school-shootings-fast-facts-dg/index.html", stream=True)

