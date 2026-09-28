from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain.agents import create_agent

from dotenv import load_dotenv
load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'http://api.weatherstack.com/current?access_key=8aa2d80c3e2c0825bc2efe223287e6c6&query={city}'

  response = requests.get(url)

  return response.json()


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    provider="auto",
    temperature=0.7,
    max_new_tokens=2048
)

model = ChatHuggingFace(llm = llm)


# create the ReAct agent
agent = create_agent(
    model = model,
    tools =[search_tool , get_weather_data],
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "Find the capital of Rajasthan, then find it's current weather condition"}
    ]
})

print(response["messages"][-1].content)
