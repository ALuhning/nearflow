from langflow.components.agents.nearai_agent import NearAIAgentComponent
from langflow.components.tools import CalculatorToolComponent, SearchAPIComponent
from langflow.components.input_output import ChatInput, ChatOutput
from langflow.graph import Graph


def simple_near_agent_graph():
    # Create components
    chat_input = ChatInput()
    
    # Create NearAI Agent component
    agent = NearAIAgentComponent()
    agent.set(
        system_prompt="You are a helpful assistant that can use tools to answer questions and perform tasks.",
        input_value=chat_input.message_response
    )
    
    # Create tools
    calculator = CalculatorToolComponent()
    search_tool = SearchAPIComponent()
    
    # Connect tools to agent
    agent.set(tools=[calculator.build_tool, search_tool.build_tool])
    
    # Create output
    chat_output = ChatOutput()
    chat_output.set(input_value=agent.message_response)
    
    return Graph(
        start=chat_input,
        end=chat_output,
        flow_name="Simple NearAI Agent",
        description="A simple agent using Near AI that can use calculator and search tools to answer questions."
    )
