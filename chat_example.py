from langchain_core.messages import ChatMessage, HumanMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(model="deepseek-r1:latest")

messages = [
    ChatMessage(role="control", content="thinking"),
    HumanMessage("What is 3^3?")
]

response = llm.invoke(messages)
print(response.content)