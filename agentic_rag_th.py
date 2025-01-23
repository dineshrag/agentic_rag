from phi.agent import Agent 
from phi.model.openai import OpenAIChat
from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.csv import CSVKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
from phi.document.chunking.fixed import FixedSizeChunking
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k
from dotenv import load_dotenv


load_dotenv()

knowledge_base = CSVKnowledgeBase(
    path="data/transaction_data_jan_feb_2024.csv",
    vector_db=LanceDb(
        table_name="transaction_data_jan_feb_2024",
        uri="tmp/lancedb/transaction_hist/openai",
        search_type=SearchType.vector,
        embedder=OpenAIEmbedder()
    ),
    chunking_strategy=FixedSizeChunking(chunk_size=1024, overlap=10)
)

knowledge_base.load()

agent_rag_imdb = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools = [DuckDuckGo(), Newspaper4k()],
    knowledge=knowledge_base,
    add_context=True,
    show_tool_calls=True,
    markdown=True,
    instructions="Please display the data in tabular format only"
)

agent_rag_imdb.print_response("Can you provide how much user spend on entertainment in January from knowledge base. Please include Grand Total at the end. ", stream=True)
agent_rag_imdb.print_response("What is recent news in USA", stream=True)
agent_rag_imdb.print_response("can summarize the article https://edition.cnn.com/us/school-shootings-fast-facts-dg/index.html", stream=True)

