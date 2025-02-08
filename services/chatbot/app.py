import os
from flask import Flask, request, jsonify
from config.settings import OPENAI_API_KEY, CHROMA_DB_PATH  # Import API key and DB path from config
from flask_restful import Api, Resource
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from services.vector_store.store import VectorStore  # Import the vector store module

# Set OpenAI API Key as an environment variable
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize Flask application
app = Flask(__name__)
api = Api(app)  # Use Flask-RESTful for API routing

# Initialize the VectorStore using ChromaDB
vector_store = VectorStore()

class Chatbot(Resource):
    """Handles user queries and returns relevant stored results."""

    def post(self):
        """
        Receives a JSON request containing a query, 
        retrieves the most relevant results from the vector database, 
        and returns them as a response.
        """
        data = request.get_json()  # Get JSON data from request
        query_text = data.get("query")  # Extract the query parameter

        # Validate the request
        if not query_text:
            return jsonify({"error": "Query cannot be empty"}), 400

        # Retrieve top 3 most relevant results
        results = vector_store.query(query_text, top_k=3)

        # Format the results into a structured response
        response = [
            {"text": res.page_content, "source": res.metadata.get("source", "N/A")}
            for res in results
        ]

        return jsonify({"results": response})

# Define the API route for the chatbot
api.add_resource(Chatbot, "/chat")

if __name__ == "__main__":
    """
    Runs the Flask application on port 5000.
    The `host="0.0.0.0"` allows external access (useful for deployment).
    """
    app.run(host="0.0.0.0", port=5000, debug=True)
