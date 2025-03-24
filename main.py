from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    
# llm = OllamaLLM(model="llama3.2:1b", format="json") 
llm = ChatOpenAI(model="gpt-3.5-turbo")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
 
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an intelligent research assistant, dedicated to providing well-structured and insightful responses.  
            Analyze the user’s query, leverage the necessary tools, and deliver a comprehensive research summary.  
            Format your response as follows and provide no additional text\n{format_instructions}
            
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = input("Give title for research? ") 
raw_response = agent_executor.invoke({"query": query, "chat_history": []})  

try:
    structured_response = parser.parse(raw_response["output"])  
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)