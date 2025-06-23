from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="vicuna:latest",
)

texts = [
    "Blue is the best color",
    "Brazil is the best country",
    "Expectations kills movies"
]

vectorstore = InMemoryVectorStore.from_texts(
    texts,
    embedding=embeddings,
)

# Use the vectorstore as a retriever
retriever = vectorstore.as_retriever()

# Retrieve the most similar text
retrieved_documents = retriever.invoke("What is bad for the movies ?")

# show the retrieved document's content
print(retrieved_documents[0].page_content)