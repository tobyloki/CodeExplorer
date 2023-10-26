from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.agents import Tool
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
from langchain.prompts import BaseChatPromptTemplate
from langchain.chains import LLMMathChain


# Set up the base template
template = """
Answer the following question as best you can. You have access to the following tools:

{tools}

Use the following format:

\nQuestion: the input question you must answer
\nThought: you should always think about what to do. if you have an answer to the question, then submit the final answer.
\nAction: the action to take, should be one of [{tool_names}]
\nAction Input: the input to the action
\nObservation: the result of the action
\n... (this Thought/Action/Action Input/Observation can repeat N times)
\nThought: I now know the final answer
\nFinal Answer: the final answer to the originals input question

Begin!

Previous conversation history:
{chat_history}

New Question: {input}
{agent_scratchpad}"""


# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            print("Could not parse LLM output, but finishing agent anyways: `{llm_output}`")
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

def get_agent_executor(_qa, llm):
    # Create agent executor and use qa and math as tools
    llm_math_chain = LLMMathChain.from_llm(llm=llm)
    tools = [
        Tool.from_function(
            name = "Code",
            func=_qa.run,
            description="useful for when you need to answer questions about code"
        ),
        Tool.from_function(
            func=llm_math_chain.run,
            name="Calculator",
            description="useful for when you need to answer questions about math"
        )
    ]
    
    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "chat_history", "intermediate_steps"]
    )
    output_parser = CustomOutputParser()
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        handle_parsing_errors=False,
        output_parser=output_parser,
        stop=["\nObservation:", "Observation:", "\nObservation", "Observation"],
        allowed_tools=tool_names,
        max_iterations=2,
    )
    # only retain the last message in the chat history
    memory = ConversationBufferWindowMemory(k=1, memory_key="chat_history", return_messages=True)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        max_iterations=2,
    )

    # agent_executor = initialize_agent(
    #     tools,
    #     llm,
    #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=True,
    #     handle_parsing_errors=False,
    #     agent_kwargs={
    #         'output_parser': output_parser
    #     },
    #     memory=memory,
    #     max_iterations=2,
    # )

    return agent_executor