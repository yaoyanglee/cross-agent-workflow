#!/usr/bin/env python
import os
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel
from typing import List
from crewai import Agent, Crew, Process, Task
import json
from agents import DeveloperAgents, SupervisorAgents
from tasks import DeveloperTask, SupervisingTask
import yaml
from dotenv import load_dotenv
from crewai import LLM

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
tasks_config = load_yaml(
    r"C:\Users\yylee\OneDrive\Desktop\synapxe\playground\langgraph\crewflow-clean\crewflow-clean\config\tasks.yaml")

# print(developer_file_management)

developer_file_management = DeveloperAgents().developer_file_management()
supervisor = SupervisorAgents().supervisor()
web_searcher = DeveloperAgents().web_searcher()

managing_files = DeveloperTask().managing_files(developer_file_management)
web_searching = DeveloperTask().web_searching(web_searcher)
supervising = SupervisingTask().supervising(supervisor)

agents_dict = {
    'developer_file_management': {"id": "developer_file_management", "configuration": agents_config['developer_file_management'], "agent": developer_file_management},
    'web_searcher': {"id": "web_searcher", "configuration": agents_config['web_searcher'], "agent": web_searcher},
}
tasks_dict = {
    'managing_files': {"id": "managing_files", "configuration": tasks_config['managing_files'], "task": managing_files},
    'web_searching': {"id": "web_searching", "configuration": tasks_config['web_searching'], "task": web_searching},
}


class State(BaseModel):
    query: str = ""  # New field to store query
    # Field to store paths of all files and folders in the current directory and subdirectories
    directory_contents: List[str] = []


class State(BaseModel):
    query: str = ""  # New field to store query
    # Field to store paths of all files and folders in the current directory and subdirectories
    directory_contents: List[str] = []

    def update_directory_contents(self, directory: str = os.getcwd()):
        """Method to update the directory_contents with all files and folders, including those in subdirectories."""
        all_files_and_folders = []
        # Traverse the directory recursively and collect file/folder paths
        for root, dirs, files in os.walk(directory):
            # Add folders
            all_files_and_folders.extend([os.path.join(root, d) for d in dirs])
            # Add files
            all_files_and_folders.extend(
                [os.path.join(root, f) for f in files])

        # Update the directory_contents field
        print(all_files_and_folders)
        self.directory_contents = all_files_and_folders


class Flow(Flow[State]):
    @start()
    def chooseAgentAndTaskToResolveUserQuestion(self):
        print("Starting the flow...")

        # self.state.update_directory_contents()
        self.state.update_directory_contents(directory="./FileSystem")

        # If query is passed, update it in the state
        if self.state.query:
            print(f"Custom query received: {self.state.query}")
        query = self.state.query
        crew = Crew(agents=[supervisor], tasks=[supervising])

        inputs = {
            "query": query,
            "agents": json.dumps({"Agents": [{"id": item["id"], "configuration": item["configuration"]} for item in agents_dict.values()]}),
            "tasks": json.dumps({"Tasks": [{"id": item["id"], "configuration": item["configuration"]} for item in tasks_dict.values()]}),
        }

        output = crew.kickoff(inputs=inputs)
        print("output.raw", output.raw)
        result = json.loads(output.raw)
        print("result:", result)
        data = {"delegation_config": result, "query": query}
        print("first", data)
        return result

    @listen(chooseAgentAndTaskToResolveUserQuestion)
    def resolveUserQuestion(self, data):
        """
        This function is responsible for resolving the user request.
        It returns the final result of the user request.
        """

        choosen_agents_list = [agents_dict[id]["agent"]
                               for id in data["agents"]]
        choosen_tasks_list = [tasks_dict[id]["task"] for id in data["tasks"]]

        crew = Crew(agents=choosen_agents_list,
                    tasks=choosen_tasks_list, planning=True, planning_llm=llm)

        output = crew.kickoff(inputs={
                              "query": self.state.query, "directory_contents": self.state.directory_contents})

        return output

    # Method to accept the query as input from the user or system
    def accept_query(self, query: str):
        print(f"Received query: {query}")
        self.state.query = query


def kickoff():
    flow = Flow()

    # Example: Pass a query to the flow
    # Customize your query
    flow.accept_query("create a folder called folder 4")
    flow.kickoff()


def plot():
    flow = Flow()

    flow.plot()


if __name__ == "__main__":
    kickoff()
