import os
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chains import LLMMathChain

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

llm = OpenAI()

search = GoogleSearchAPIWrapper()
math_chain = LLMMathChain(llm=llm)

tools = [
    Tool(name="Search", func=search.run, description="Search the web"),
    Tool(name="Calculator", func=math_chain.run, description="Math calculations"),
]

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

while True:
    q = input("\nTask: ")
    if q.lower() == "exit":
        break
    print(agent.run(q))
