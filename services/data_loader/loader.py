from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup

class DataLoader:
    def __init__(self, url, max_depth=1): #setting max depth to 1(how deeply it scrapes links)
        self.url = url
        self.max_depth = max_depth

    def extract_text(self, html_content): #Extracts clean text from HTML using BeautifulSoup.
       
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(separator="\n", strip=True) #Leading and trailing spaces are removed

    def load_data(self): #Loads data from the given URL using LangChain's RecursiveUrlLoader.
        
        loader = RecursiveUrlLoader(
            url=self.url,
            max_depth=self.max_depth,
            extractor=self.extract_text
        )
        documents = loader.load()
        return documents

if __name__ == "__main__":
    url = "https://brainlox.com/courses/category/technical"
    data_loader = DataLoader(url, max_depth=2)
    docs = data_loader.load_data()
    
    # Print the first few extracted documents
    for i, doc in enumerate(docs[:3]):
        print(f"\nDocument {i+1}:\n{doc.page_content}\n{'-'*50}")
