"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast
import json
from dotenv import load_dotenv
import uuid

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model

from react_agent.marketing_posts.main import run
from react_agent.crewai_agents.lead_score_flow.main import kickoff

load_dotenv()

# Defining an output structure


class SupervisorResponse(BaseModel):
    next_step: str


class AgentResponse(BaseModel):
    content: str
    next_step: str

# Define the function that calls the model


async def supervisor_agent(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """

    configuration = Configuration.from_runnable_config(config)

    # Initialize the model with tool binding. Change the model or add more tools here.
    # model = load_chat_model(configuration.model).bind_tools(TOOLS)
    model = load_chat_model(configuration.model)
    model = model.with_structured_output(SupervisorResponse)

    # Format the system prompt. Customize this to change the agent's behavior.
    # Get the correct prompt for this agent
    system_message = configuration.get_prompt("supervisor_agent")

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message},
                *state.messages], config
        ),
    )

    print("\n############## SUPERVISOR STATE ##################\n", state)

    print("\n ############## SUPERVISOR RESPONSE ############ \n",
          response, "   ", type(response))

    # Handle the case when it's the last step and the model still wants to use a tool
    # if state.is_last_step and response.tool_calls:
    #     return {
    #         "messages": [
    #             AIMessage(
    #                 id=response.id,
    #                 content="Sorry, I could not find an answer to your question in the specified number of steps.",
    #             )
    #         ]
    #     }

    if state.is_last_step:
        return {
            "messages": [
                AIMessage(
                    id=uuid.uuid4,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Parse the structured JSON response from the model
    try:
        # structured_output = json.loads(response.content)
        # print("\n++++++++ STRUCTURED OUTPUT +++++++++++++\n", structured_output)
        # next_step = structured_output['next_step']

        if response.next_step:
            response = response.model_dump_json()
            return {"messages": [AIMessage(id=uuid.uuid4, content=response)]}

    except (json.JSONDecodeError, KeyError):
        # Default to end if parsing fails or if the output is malformed
        return {"messages": [AIMessage(id=uuid.uuid4, content={'content': 'JSON Key Error', 'next_step': '__end__'})]}

    # Fallback to return the response if no next step is defined
    return {"messages": [AIMessage(id=uuid.uuid4, content=response)]}


async def researcher_agent(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_runnable_config(config)

    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.get_prompt("researcher_agent")

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages], config
        ),
    )

    print("\n######## RESEARCHER AGENT RESPONSE ############\n", response)
    # Handle the case when it's the last step and the model still wants to use a tool
    # if state.is_last_step and response.tool_calls:
    #     return {
    #         "messages": [
    #             AIMessage(
    #                 id=response.id,
    #                 content="Sorry, I could not find an answer to your question in the specified number of steps.",
    #             )
    #         ]
    #     }

    if state.is_last_step:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


async def summary_agent(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_runnable_config(config)

    # Initialize the model with tool binding. Change the model or add more tools here.
    # model = load_chat_model(configuration.model).bind_tools(TOOLS)
    model = load_chat_model(configuration.model)

    # Format the system prompt. Customize this to change the agent's behavior.
    # system_message = configuration.system_prompt.format(
    #     system_time=datetime.now(tz=timezone.utc).isoformat()
    # )
    system_message = configuration.get_prompt("summary_agent")

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages], config
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    # if state.is_last_step and response.tool_calls:
    #     return {
    #         "messages": [
    #             AIMessage(
    #                 id=response.id,
    #                 content="Sorry, I could not find an answer to your question in the specified number of steps.",
    #             )
    #         ]
    #     }

    if state.is_last_step:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


async def marketing_agent(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    # configuration = Configuration.from_runnable_config(config)

    # # Initialize the model with tool binding. Change the model or add more tools here.
    # model = load_chat_model(configuration.model).bind_tools(TOOLS)
    # # model = load_chat_model(configuration.model)

    # # Format the system prompt. Customize this to change the agent's behavior.
    # system_message = configuration.get_prompt("researcher_agent")

    crew_input = {"langgraph_messages": state.messages}
    crew_output = run(crew_input)
    print("########## CREW OUTPUT ############", crew_output)

    # Constructing a response that is compatible with the langgraph nodes return values
    response = AIMessage(
        # You can generate a unique ID if needed
        id=f"crewai-marketing-run-{uuid.uuid4()}",
        content=crew_output,  # Use CrewAI's generated content
        additional_kwargs={},  # Empty metadata (can be extended)
        response_metadata={
            "source": "CrewAI-marketing",
            "model_name": "CrewAI-marketing-Generated",
            "finish_reason": "complete",
        },
    )

    if state.is_last_step:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # # Return the model's response as a list to be added to existing messages
    # return {"messages": [response]}
    return {"messages": [response]}


async def lead_agent(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    # configuration = Configuration.from_runnable_config(config)

    # # Initialize the model with tool binding. Change the model or add more tools here.
    # model = load_chat_model(configuration.model).bind_tools(TOOLS)
    # # model = load_chat_model(configuration.model)

    # # Format the system prompt. Customize this to change the agent's behavior.
    # system_message = configuration.get_prompt("researcher_agent")
    print("\n######### LEAD AGENT MESSAGES ############\n",
          state.messages)

    crew_input = {"job_description": state.messages}
    crew_output = await kickoff(crew_input)
    print("########## CREW OUTPUT ############", crew_output)

    # Constructing a response that is compatible with the langgraph nodes return values
    response = AIMessage(
        # You can generate a unique ID if needed
        id=f"crewai-lead-generation-run-{uuid.uuid4()}",
        content=crew_output,  # Use CrewAI's generated content
        additional_kwargs={},  # Empty metadata (can be extended)
        response_metadata={
            "source": "CrewAI-lead",
            "model_name": "CrewAI-lead-Generated",
            "finish_reason": "complete",
        },
    )

    if state.is_last_step:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # # Return the model's response as a list to be added to existing messages
    # return {"messages": [response]}
    return {"messages": [response]}

# Define a new graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

builder.add_node(researcher_agent)
builder.add_node(supervisor_agent)
builder.add_node(summary_agent)
builder.add_node(marketing_agent)
builder.add_node(lead_agent)

# Define the two nodes we will cycle between
# builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as `call_model`
# This means that this node is the first one called
# builder.add_edge("__start__", "call_model")
builder.add_edge("__start__", "supervisor_agent")
builder.add_edge("summary_agent", "supervisor_agent")
builder.add_edge("marketing_agent", "supervisor_agent")
builder.add_edge("lead_agent", "supervisor_agent")

# The researcher conditionally routes back to the supervisor node or tools node depending on the situation


def researcher_routing(state: State) -> Literal["tools", "supervisor_agent"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    print("\n ############ RESEARCHER LAST MSG ##################\n", last_message)
    print("\n ############ RESEARCHER TOOL CALL ##################\n",
          last_message.tool_calls)
    print("\n############## RESEARCHER ROUTING STATE MESSAGES ##################\n", state.messages)
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "supervisor_agent"
    # Otherwise we execute the requested actions
    return "tools"


def supervisor_routing(state: State) -> Literal["__end__", "summary_agent", "researcher_agent", "marketing_agent", "lead_agent"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    print("\n############## SUPERVISOR ROUTING LAST MSG #################\n", last_message)
    print("\n############## STATE MESSAGES ##################\n", state.messages)
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )

    # Route based on where the supervisor think we should go
    return last_message.content


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "researcher_agent",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    researcher_routing,
)

# Adding conditional edge for supervisor routing
builder.add_conditional_edges("supervisor_agent", supervisor_routing)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "researcher_agent")

# Compile the builder into an executable graph
# You can customize this by adding interrupt points for state updates
graph = builder.compile(
    interrupt_before=[],  # Add node names here to update state before they're called
    interrupt_after=[],  # Add node names here to update state after they're called
)
graph.name = "ReAct Agent"  # This customizes the name in LangSmith
