import os
import chromadb
from config.settings import CHROMA_DB_PATH  # Import path from settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from services.data_loader.loader import DataLoader  # Import from loader.py

# Load OpenAI API Key from config
from config.settings import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class VectorStore:
    def __init__(self):
        """Initialize vector store with ChromaDB using config path."""
        self.persist_directory = CHROMA_DB_PATH  # Use path from config
        self.embedding_function = OpenAIEmbeddings()
        self.vector_db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function
        )

    def add_documents(self, documents):
        """Add text documents to the vector database."""
        texts = [doc.page_content for doc in documents]
        metadatas = [{"source": doc.metadata["source"]} for doc in documents]
        self.vector_db.add_texts(texts, metadatas=metadatas)

        # Persist changes
        self.vector_db.persist()
        print("✅ Data stored in vector database successfully!")

    def query(self, query_text, top_k=3):
        """Retrieve the most relevant documents based on a query."""
        results = self.vector_db.similarity_search(query_text, k=top_k)
        return results

if __name__ == "__main__":
    # Load data
    url = "https://brainlox.com/courses/category/technical"
    data_loader = DataLoader(url, max_depth=2)
    documents = data_loader.load_data()

    # Store embeddings
    vector_store = VectorStore()
    vector_store.add_documents(documents)

    # Test Query
    query_text = "Python courses"
    results = vector_store.query(query_text)

    # Print results
    for i, res in enumerate(results):
        print(f"\nResult {i+1}:\n{res.page_content}\n{'-'*50}")
