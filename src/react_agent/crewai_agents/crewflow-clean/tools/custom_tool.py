import os
import pandas as pd
import json
import subprocess
from typing import Type, List
from crewai.tools import BaseTool, tool
from pydantic import BaseModel, Field
#from rapidfuzz import process
import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

# Retrieve environment variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Debugging: Print to check if values are loaded (remove in production)
if AZURE_OPENAI_ENDPOINT is None or AZURE_OPENAI_API_KEY is None:
    raise ValueError("Environment variables are not set! Check .env or system variables.")

# Initialize LLM
llm = LLM(
    model="azure/gpt-4o-mini",
    base_url=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY
)
# ----------- Utility Functions ----------- #


class ReadFileToolInput(BaseModel):
    """Input schema for ReadFileTool."""
    filepath: str = Field(..., description="Path to the file to be read.")


class ReadFileTool(BaseTool):
    """Tool to read the contents of a file based on its extension. Does not read folders"""
    name: str = "Read File Tool"
    description: str = "Reads a file (.csv, .xls, .xlsx, .json, .txt) and returns its content."
    args_schema: Type[BaseModel] = ReadFileToolInput

    def _run(self, filepath: str) -> str:
        if not os.path.exists(filepath):
            return f"Error: File '{filepath}' not found."

        _, ext = os.path.splitext(filepath)
        try:
            if ext in ['.csv']:
                return pd.read_csv(filepath).to_dict()
            elif ext in ['.xls', '.xlsx']:
                return pd.read_excel(filepath).to_dict()
            elif ext in ['.json']:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif ext in ['.txt']:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                return f"Unsupported file format: {ext}"
        except Exception as e:
            return f"Error reading file: {str(e)}"


class GenerateCodeToolInput(BaseModel):
    """Input schema for GenerateCodeTool."""
    query: str = Field(..., description="Task description for which Python code should be generated.")
    directory_contents:str  = Field(..., description="Directory content of the working directory.Any manipulation should only be doen in the FileSystem folder")


class GenerateCodeTool(BaseTool):
    """Tool to generate Python code based on a given task. """
    name: str = "Generate Code Tool"
    description: str = "Generates Python code for a given task description."
    args_schema: Type[BaseModel] = GenerateCodeToolInput

    def _run(self, query: str, directory_contents:str) -> str:
        # Ensure directory state is up-to-date
 
        prompt = f"""
        Write Python code for the following task:
        
        {query}

        Any manuplutaion should only be done in the FileSystem folder {directory_contents}. The base directory should always be /FileSystem folder. 
        The generated code should be relevant to the existing files and folders. Code should only limit to the FileSystem folder and and subfolders.
        Output only the code.
        """
        
        output = llm.call(prompt)  
        return output



class ExecuteCodeToolInput(BaseModel):
    """Input schema for ExecuteCodeTool."""
    code: str = Field(..., description="Python code as a string to be executed.")


class ExecuteCodeTool(BaseTool):
    """Tool to execute Python code using subprocess."""
    name: str = "Execute Code Tool"
    description: str = "Executes a given Python code string and returns the output."
    args_schema: Type[BaseModel] = ExecuteCodeToolInput

    def _run(self, code: str) -> str:
        try:
            result = subprocess.run(["python", "-c", code], capture_output=True, text=True, timeout=5)
            return result.stdout or result.stderr
        except subprocess.TimeoutExpired:
            return "Error: The code execution exceeded the time limit."
        except Exception as e:
            return f"Error: An unexpected error occurred during execution. {str(e)}"



# ----------- Tools ----------- for #

