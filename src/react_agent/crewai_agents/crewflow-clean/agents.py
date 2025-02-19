from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai import LLM

# Adjust import path if needed
from tools import ReadFileTool, GenerateCodeTool, ExecuteCodeTool

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
import os
from dotenv import load_dotenv
from crewai import LLM
import yaml
# Load environment variables from .env file (if using dotenv)
load_dotenv()

# Retrieve environment variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
# Debugging: Print to check if values are loaded (remove in production)
if AZURE_OPENAI_ENDPOINT is None or AZURE_OPENAI_API_KEY is None:
    raise ValueError(
        "Environment variables are not set! Check .env or system variables.")

# Initialize LLM
llm = LLM(
    model="azure/gpt-4o-mini",
    base_url=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY
)


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


agents_config = load_yaml(
    r"C:\Users\yylee\OneDrive\Desktop\synapxe\playground\langgraph\crewflow-clean\crewflow-clean\config\agents.yaml")


class DeveloperAgents:

    @agent
    def developer_file_management(self) -> Agent:
        return Agent(
            config=agents_config["developer_file_management"],
            verbose=True,
            llm=llm,
            tools=[ReadFileTool(), GenerateCodeTool(), ExecuteCodeTool()],
        )

    @agent
    def web_searcher(self) -> Agent:
        return Agent(
            config=agents_config["web_searcher"],
            verbose=True,
            llm=llm,
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
        )


class SupervisorAgents:

    @agent
    def supervisor(self) -> Agent:
        return Agent(
            config=agents_config["supervisor"],
            verbose=True,
            llm=llm
        )
