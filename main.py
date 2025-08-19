import json
import os
from copy import deepcopy
from typing import Literal, Union

import instructor
import openai
from anthropic import Anthropic
from anthropic.types import ToolUseBlock, TextBlock, Message
from dotenv import load_dotenv
from ollama import chat, ChatResponse
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.responses import Response
from pydantic import BaseModel


# format type should be Pydantic Model
class JokeType(BaseModel):
    joke: str

class WeatherType(BaseModel):
    city: str
    degree: int
    unit: Literal["celsius","fahrenheit"]
    weather_text: str

class ToolResponse(BaseModel):
    tool_call_id: str
    tool_response: str

def setup():
    load_dotenv()

def get_weather(city: str, format) -> str:
    """
    Tool for getting weather for particular city
    :param city: City for which to get weather
    :return: return weathers string
    """
    return f"It is sunny in {city}, it is 20 degree {format}"

def add_two_numbers(number1: str, number2: str) -> str:
    """
    Tool for adding two numbers together
    :param number1: First number
    :param number2: Second number
    :return: Result of adding two numbers together
    """
    return str(float(number1) + float(number2))

TOOLS = [
        {
            # type not required in anthropic
            "type": "function",
            # Add the data below in "function": {} for ollama
            "name": "get_weather",
            "description": "Get the current weather for a city",
            # this should be "input_schema" at place for "parameters"
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city for which to get weather",
                    },
                    "format": {
                        "type": "string",
                        "description": "The format to return the weather in, e.g. 'celsius' or 'fahrenheit'",
                    }
                },
                "required": ["city", "format"],
            }
        },
        {
            "type": "function",
            "name": "add_two_numbers",
            "description": "Add two numbers together",
            "parameters": {
                "type": "object",
                "properties": {
                    "number1": {
                        "type": "string",
                        "description": "First number to be added.",
                    },
                    "number2": {
                        "type": "string",
                        "description": "Second number to be added.",
                    }
                },
                "required": ["number1", "number2"],
            }
        },
    ]

TOOLS_LIST = {
    "get_weather": get_weather,
    "add_two_numbers": add_two_numbers,
}

def get_messages():
    pass

def add_message():
    pass

def call_ollama(model, messages, tools, format)-> ChatResponse:
    ollama_tools = []
    for tool in tools:
        t = deepcopy(tool)
        ollama_tool = {}
        ollama_tool = {
            "type": t.pop("type"),
            "function": t,
        }
        ollama_tools.append(ollama_tool)
    if format is None or len(tools) > 0:
        response = chat(
            model=model.replace("ollama:", ""),
            messages=messages,
            tools=ollama_tools,
            options={'temperature': 0},
        )
        return response
    response = chat(
        model=model.replace("ollama:", ""),
        messages=messages,
        format=format.model_json_schema(),
        options={'temperature': 0}
    )
    return response

def call_openai(model:str, messages, tools, format):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI()
    if format is None or len(tools) > 0:
        response = client.responses.create(
            model=model.replace("openai:", ""),
            input=messages,
            tools=tools,
        )
        if not format is None:
            for item in response.output:
                if item.type != "function_call":
                    # Todo Can I skip this by if else, so that i can save a openai_call
                    # Not truly since, it would still be a call, I can make it a bit readable, but for tool_calls with structured output, it is two calls.
                    new_message = [{
                        "role": "user",
                        "content": f"Convert the given response into given format. <response>{response.output_text}</response> and <format>{format.model_json_schema()}</format>",
                    }]
                    completition = client.chat.completions.parse(
                        model=model.replace("openai:", ""),
                        messages=new_message,
                        response_format=format,
                    )
                    return completition
        return response

    completion = client.chat.completions.parse(
        model=model.replace("openai:", ""),
        messages=messages,
        response_format=format,
    )
    return completion


def call_anthropic(model:str, messages, tools, format):
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    anthropic_client = instructor.from_anthropic(client=Anthropic(api_key=anthropic_api_key,))

    anthropic_tools = []
    for tool in tools:
        anthropic_tool = {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"],
        }
        anthropic_tools.append(anthropic_tool)

    anthropic_messages = []
    for message in messages:
        if message["role"] == "system":
            message = {
                "role": "user",
                "content": f"{message['content']}"
            }
            anthropic_messages.append(message)
        anthropic_messages.append(message)
    if format is None or len(tools) > 0:
        client = Anthropic(
            api_key=anthropic_api_key,
        )
        message = client.messages.create(
            model=model.replace("anthropic:", ""),
            max_tokens=1024,
            messages=anthropic_messages,
            tools=anthropic_tools,
        )
        # Tool Call response is
        # Message(id='msg_011zLVY8k2Q5D3pYHarfkexV', content=[ToolUseBlock(id='toolu_01LQrksbDWe2WieXUNdTBhMV', input={'city': 'Winnipeg', 'format': 'celsius'}, name='get_weather', type='tool_use')], model='claude-3-haiku-20240307', role='assistant', stop_reason='tool_use', stop_sequence=None, type='message', usage=Usage(cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=404, output_tokens=72, server_tool_use=None, service_tier='standard'))
        # TODO
        if not format is None:
            for block in message.content:
                if not isinstance(block, ToolUseBlock):
                    new_message = [{
                        "role": "user",
                        "content": f"Convert the given response into given format. <response>{message.content[0].text}</response> and <format>{format.model_json_schema()}</format>",
                    }]
                    response = anthropic_client.messages.create(
                        model=model.replace("anthropic:", ""),
                        messages=new_message,
                        max_tokens=1024,
                        response_model=format,
                    )
                    return response
        return message

    response = anthropic_client.messages.create(
        model=model.replace("anthropic:", ""),
        messages=anthropic_messages,
        max_tokens=1024,
        response_model=format,
    )
    return response

def call_llm(model:str, messages, tools=[], format=None):
    """
    An abstraction for all llms.
    Options are OpenAI, anthropic, ollama, and possibly more.
    :param model: Model to be called
    :param messages: List of messages that is fed to llm.
    :param format: Response should be in this format
    :param tools: List of tools that llm has access to.
    """
    fake_none_format = None
    # if len(tools) > 0:
    #     format = None
    if "openai" in model.lower():
        print(f"Calling OpenAI {model}")
        response: Union[Response,ChatCompletion] =  call_openai(model=model, messages=messages, tools=tools, format=format)
        if not (tools is  None or len(tools) == 0):
            if hasattr(response, "output") and len(response.output) > 0:
                for tool_call in response.output:
                    if tool_call.type == "function_call":
                        # OpenAI tool call is
                        # ResponseFunctionToolCall(arguments='{"city":"Toronto","format":"celsius"}', call_id='call_AC5hTmTP4COCP46JrEDruV04', name='get_weather', type='function_call', id='fc_6897fa77316c819596ddaed46caa92cc09365609b1078dd1', status='completed')
                        # of type
                        # <class 'ollama._types.Message.ToolCall'>

                        # print(f"OpenAI's tool call type is  {type(tool_call)}")
                        print(f"Calling {tool_call.name}")
                        tool_response = call_tool(tool_name=tool_call.name, arguments=json.loads(tool_call.arguments), tool_call_id=tool_call.id)
                        # print(f"OpenAI's tool call response is  {tool_response}")
                        tool_message = {
                            "role": "user",
                            "content": f"{tool_call.name} tool's response is {str(tool_response.tool_response)}"
                        }
                        next_message = messages + [tool_message]
                        response: Response = call_openai(model=model, messages=next_message, tools=[], format=format)
                        # print(response)
        if format is None:
            return response.output_text
        completion = response
        # print("Before parsing: ", completion)
        message = completion.choices[0].message
        if message.parsed:
            return message.parsed
        else:
            print(message.refusal)
            raise

    elif "ollama" in model.lower():
        print(f"Calling ollama {model}")
        response: ChatResponse =  call_ollama(model, messages, tools, format)
        # print(f"Ollama's initial response is  {response}")
        if not (tools is  None or len(tools) == 0):
            if hasattr(response.message, "tool_calls") and len(response.message.tool_calls) > 0:
                for tool_call in response.message.tool_calls:
                    # Ollama's tool call is
                    # function=Function(name='get_weather', arguments={'city': 'Toronto', 'format': 'celsius'})
                    # of type
                    # <class 'openai.types.responses.response_function_tool_call.ResponseFunctionToolCall'>

                    # print(f"Ollama is calling tool {tool_call.function.name}")
                    print(f"Calling {tool_call.function.name}")
                    tool_response = call_tool(tool_name=tool_call.function.name, arguments=tool_call.function.arguments,tool_call_id="OLLAMA_ID")
                    # print(f"Ollama's tool call response is  {tool_response}")
                    # print()
                    tool_message = {
                        "role": "tool",
                        "content" : f"{tool_call.function.name} tool's response is {str(tool_response.tool_response)}",
                        "tool_call_id": tool_response.tool_call_id,
                    }
                    next_message = messages + [tool_message]
                    response: ChatResponse = call_ollama(model=model, messages=next_message,tools=[], format=format)
                    # print(f"Final Response: {response}")
        if format is None:
            return response.message.content
        else:
            answer = format.model_validate_json(response.message.content)
            return answer

    elif "anthropic" in model.lower():
        print(f"Calling anthropic {model}")
        message: Message = call_anthropic(model, messages, tools, format)
        if not (tools is  None or len(tools) == 0):
            if hasattr(message, "content") and len(message.content) > 0:
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        # print(f"Anthropic's tool call type is  {type(block)}")
                        # print(f"Anthropic's tool call response is  {block}")
                        print(f"Calling {block.name}")
                        tr = call_tool(tool_name=block.name, arguments=block.input, tool_call_id=block.id)
                        # print(f"Anthropic tool call response is  {tr}")
                        tool_message = {
                            "role": "user",
                            "content": f"{block.name} tool's response is {str(tr.tool_response)}"
                        }
                        next_message = messages + [tool_message]
                        message: Message = call_anthropic(model=model, messages=next_message, tools=[], format=format)
        if format is None:
            for block in message.content:
                if isinstance(block, TextBlock):
                    return block.text
            return message
        return message

    else:
        print(f"Yet to implement llm: {model}")
        default_model="openai:gpt-4o-mini"
        print(f"using default llm: {default_model}")
        return call_llm(model=default_model, messages=messages, tools=tools, format=format)

def call_tool(tool_name, arguments, tool_call_id) -> ToolResponse :
    """
    Helper function to call tools and return tool responses.
    :param tool_name: name of tool to be called
    :param arguments: arguments passed to function
    :param tool_call_id: id of tool call
    :return: tool response
    """
    if tool_name in TOOLS_LIST:
        tool_response = TOOLS_LIST[tool_name](**arguments)
    else:
        raise Exception(f"Unknown tool {tool_name}")

    return ToolResponse(
        tool_call_id=tool_call_id,
        tool_response=tool_response,
    )

if __name__ == "__main__":
    setup()

    messages = [
        {
            'role': 'system',
            'content': f'You are a helpful AI agent.',
        },
        {
            'role': 'user',
            'content': f'Tell me a joke',
        }
    ]

    # This is a general type how tools should be defined
    # But ollama and anthropic work differently

    model = "openai:gpt-4o-mini"
    response = call_llm(model=model, messages=messages, tools=[], format=None)
    print(type(response))
    print(response,"\n\n")
    #
    model = "anthropic:claude-3-haiku-20240307"
    response = call_llm(model=model, messages=messages, tools=[], format=None)
    print(type(response))
    print(response,"\n\n")

    model = "ollama:qwen3:30b"
    response = call_llm(model=model, messages=messages, tools=[], format=None)
    print(type(response))
    print(response,"\n\n")

    messages = [
        {
            'role': 'system',
            'content': f'You are a helpful AI agent.',
        },
        {
            'role': 'user',
            'content': f'Tell me current weather for Toronto',
        }
    ]

    model = "openai:gpt-4o-mini"
    response = call_llm(model=model, messages=messages, tools=TOOLS, format=None)
    print(type(response))
    print(response, "\n\n")
    #
    model = "anthropic:claude-3-haiku-20240307"
    response = call_llm(model=model, messages=messages, tools=TOOLS, format=None)
    print(type(response))
    print(response, "\n\n")

    model = "ollama:qwen3:30b"
    response = call_llm(model=model, messages=messages, tools=TOOLS, format=None)
    print(type(response))
    print(response, "\n\n")

    messages = [
        {
            'role': 'system',
            'content': f'You are a helpful AI agent.',
        },
        {
            'role': 'user',
            'content': f'Tell me current weather for Toronto',
        }
    ]

    model = "openai:gpt-4o-mini"
    response = call_llm(model=model, messages=messages, tools=TOOLS, format=WeatherType)
    print(type(response))
    print(response, "\n\n")
    #
    model = "anthropic:claude-3-haiku-20240307"
    response = call_llm(model=model, messages=messages, tools=TOOLS, format=WeatherType)
    print(type(response))
    print(response, "\n\n")

    model = "ollama:qwen3:30b"
    response = call_llm(model=model, messages=messages, tools=TOOLS, format=WeatherType)
    print(type(response))
    print(response, "\n\n")

    messages = [
        {
            'role': 'system',
            'content': f'You are a helpful AI agent.',
        },
        {
            'role': 'user',
            'content': f'Tell me a joke',
        }
    ]

    model = "openai:gpt-4o-mini"
    response = call_llm(model=model, messages=messages, tools=[], format=JokeType)
    print(type(response))
    print(response, "\n\n")
    #
    model = "anthropic:claude-3-haiku-20240307"
    response = call_llm(model=model, messages=messages, tools=[], format=JokeType)
    print(type(response))
    print(response, "\n\n")

    model = "ollama:qwen3:30b"
    response = call_llm(model=model, messages=messages, tools=[], format=JokeType)
    print(type(response))
    print(response, "\n\n")
