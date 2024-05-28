from langchain_openai import ChatOpenAI 

llm = ChatOpenAI()

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant, Start each response with 'I have read the system message'"),
    ("user", "{input}")
])

chain = prompt | llm 

chain.invoke()