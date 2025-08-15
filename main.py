import json
import os
from dotenv import load_dotenv
import openai
from openai import OpenAI
from openai.types.responses import Response
from anthropic import Anthropic
from anthropic.types.message import Message
import instructor
from pydantic import BaseModel
from ollama import chat, ChatResponse

# format type should be Pydantic Model
class JokeType(BaseModel):
    joke: str

class WeatherType(BaseModel):
    city: str
    format: str

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
    ]

TOOLS_LIST = {
    "get_weather": get_weather,
}

def get_messages():
    pass

def add_message():
    pass

def call_ollama(model, messages, tools, format)-> ChatResponse:
    ollama_tools = []
    for tool in tools:
        ollama_tool = {}
        ollama_tool["type"] = tool["type"]
        del tool["type"]
        ollama_tool = {
            "type": ollama_tool["type"],
            "function": tool,
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
    answer = format.model_validate_json(response.message.content)
    return answer

def call_openai(model:str, messages, tools, format):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI()
    if format is None or len(tools) > 0:
        response = client.responses.create(
            model=model.replace("openai:", ""),
            input=messages,
            tools=tools,
        )
        return response
    completion = client.chat.completions.parse(
        model=model.replace("openai:", ""),
        messages=messages,
        response_format=format,
    )
    message = completion.choices[0].message
    if message.parsed:
        return message.parsed
    else:
        print(message.refusal)
        raise

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
        response = client.messages.create(
            model=model.replace("anthropic:", ""),
            max_tokens=1024,
            messages=anthropic_messages,
            tools=anthropic_tools,
        )
        # Tool Call response is
        # Message(id='msg_011zLVY8k2Q5D3pYHarfkexV', content=[ToolUseBlock(id='toolu_01LQrksbDWe2WieXUNdTBhMV', input={'city': 'Winnipeg', 'format': 'celsius'}, name='get_weather', type='tool_use')], model='claude-3-haiku-20240307', role='assistant', stop_reason='tool_use', stop_sequence=None, type='message', usage=Usage(cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=404, output_tokens=72, server_tool_use=None, service_tier='standard'))
        return response


    response = anthropic_client.messages.create(
        model=model.replace("anthropic:", ""),
        messages=anthropic_messages,
        max_tokens=1024,
        response_model=format,
    )
    return response

def call_llm(model:str, `messages, tools=[], format=None):
    """
    An abstraction for all llms.
    Options are OpenAI, anthropic, ollama, and possibly more.
    :param model: Model to be called
    :param messages: List of messages that is fed to llm.
    :param format: Response should be in this format
    :param tools: List of tools that llm has access to.
    """
    if len(tools) > 0:
        format = None
    if "openai" in model.lower():
        print(f"Calling OpenAI {model}")
        response: Response =  call_openai(model=model, messages=messages, tools=tools, format=format)
        if hasattr(response, "output") and len(response.output) > 0:
            for tool_call in response.output:
                # TODO
                # OpenAI tool call is
                # ResponseFunctionToolCall(arguments='{"city":"Toronto","format":"celsius"}', call_id='call_AC5hTmTP4COCP46JrEDruV04', name='get_weather', type='function_call', id='fc_6897fa77316c819596ddaed46caa92cc09365609b1078dd1', status='completed')
                # of type
                # <class 'ollama._types.Message.ToolCall'>
                print(f"OpenAI's tool call type is  {type(tool_call)}")
                tool_response = call_tool(tool_name=tool_call.name, arguments=json.loads(tool_call.arguments), tool_call_id=tool_call.id)
                print(f"OpenAI's tool call response is  {tool_response}")
        return response

    elif "ollama" in model.lower():
        print(f"Calling ollama {model}")
        response: ChatResponse =  call_ollama(model, messages, tools, format)
        if hasattr(response.message, "tool_calls") and len(response.message.tool_calls) > 0:
            for tool_call in response.message.tool_calls:
                # TODO
                # Ollama's tool call is
                # function=Function(name='get_weather', arguments={'city': 'Toronto', 'format': 'celsius'})
                # of type
                # <class 'openai.types.responses.response_function_tool_call.ResponseFunctionToolCall'>
                print(f"Ollama's tool call type is {type(tool_call)}")
                tool_response = call_tool(tool_name=tool_call.function.name, arguments=tool_call.function.arguments,tool_call_id="OLLAMA_ID")
                print(f"Ollama's tool call response is  {tool_response}")
                print()
                response: ChatResponse = call_ollama(model=model, messages=messages.append({"role":"tool","content":f"{str(tool_response)}"}),tools=[], format=None)
                print(f"Final Response: {response}")
        return response

    elif "anthropic" in model.lower():
        print(f"Calling anthropic {model}")
        message: Message = call_anthropic(model, messages, tools, format)
        if hasattr(message, "content") and len(message.content) > 1:
            tool_call = message.content[1]
            # TODO
            # Anthropic's tool call is
            # ToolUseBlock(id='toolu_01FaQ9fsNiJUfMyTKETE9V2i', input={'city': 'Toronto', 'format': 'celsius'}, name='get_weather', type='tool_use')
            # of type
            # <class 'anthropic.types.tool_use_block.ToolUseBlock'>
            print(f"Anthropic's tool call type is {type(tool_call)}")
            tool_response = call_tool(tool_name=tool_call.name, arguments=tool_call.input, tool_call_id=tool_call.id)
            print(f"Anthropic's tool call response is  {tool_response}")

        return message

    else:
        print(f"Yet to implement llm: {model}")
        default_model="openai:gpt-4o-mini"
        print(f"using default llm: {default_model}")
        return call_openai(model=default_model, messages=messages, tools=tools, format=format)

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
            'content': f'Tell me current weather for Toronto in Celsius',
        }
    ]

    # This is a general type how tools should be defined
    # But ollama and anthropic work differently

    # model = "openai:gpt-4o-mini"
    # response = call_llm(model=model, messages=messages, tools=TOOLS, format=None)
    # print(type(response))
    # print(response)

    # model = "anthropic:claude-3-haiku-20240307"
    # response = call_llm(model=model, messages=messages, tools=TOOLS, format=None)
    # print(type(response))
    # print(response)
    #
    model = "ollama:qwen3:30b"
    response = call_llm(model=model, messages=messages, tools=TOOLS, format=None)
    print(type(response))
    print(response)


# Text Structured Responses

# Calling OpenAI openai:gpt-4o-mini
# <class '__main__.JokeType'>
# joke='Why did the scarecrow win an award? Because he was outstanding in his field!'

# Calling anthropic anthropic:claude-3-haiku-20240307
# <class '__main__.JokeType'>
# joke="Why can't you trust atoms? They make up everything."

# Calling ollama ollama:qwen3:30b
# <class '__main__.JokeType'>
# joke="Why don't scientists trust atoms? Because they make up everything!"


# Tool Calls Responses

# Calling OpenAI openai:gpt-4o-mini
# <class 'openai.types.responses.response.Response'>
# Response(id='resp_6897e12f21188194a83a26d08b0227c509fd49f4bbcdcb3f', created_at=1754784047.0, error=None, incomplete_details=None, instructions=None, metadata={}, model='gpt-4o-mini-2024-07-18', object='response', output=[ResponseFunctionToolCall(arguments='{"city":"Toronto","format":"celsius"}', call_id='call_GAkL1JO0ztPIe6bVvfEKV1ZU', name='get_weather', type='function_call', id='fc_6897e1308cd881948461b7ca30fea83809fd49f4bbcdcb3f', status='completed')], parallel_tool_calls=True, temperature=1.0, tool_choice='auto', tools=[FunctionTool(name='get_weather', parameters={'type': 'object', 'properties': {'city': {'type': 'string', 'description': 'The name of the city for which to get weather'}, 'format': {'type': 'string', 'description': "The format to return the weather in, e.g. 'celsius' or 'fahrenheit'"}}, 'required': ['city', 'format'], 'additionalProperties': False}, strict=True, type='function', description='Get the current weather for a city')], top_p=1.0, background=False, max_output_tokens=None, max_tool_calls=None, previous_response_id=None, prompt=None, prompt_cache_key=None, reasoning=Reasoning(effort=None, generate_summary=None, summary=None), safety_identifier=None, service_tier='default', status='completed', text=ResponseTextConfig(format=ResponseFormatText(type='text'), verbosity='medium'), top_logprobs=0, truncation='disabled', usage=ResponseUsage(input_tokens=97, input_tokens_details=InputTokensDetails(cached_tokens=0), output_tokens=20, output_tokens_details=OutputTokensDetails(reasoning_tokens=0), total_tokens=117), user=None, store=True)

# Calling anthropic anthropic:claude-3-haiku-20240307
# <class 'anthropic.types.message.Message'>
# Message(id='msg_01R5rygw38fdvK5jaqbuvW8x', content=[TextBlock(citations=None, text='Okay, let me check the current weather for Toronto in Celsius:', type='text'), ToolUseBlock(id='toolu_01DM4uMk6DQXfnDzVhp5VADv', input={'city': 'Toronto', 'format': 'celsius'}, name='get_weather', type='tool_use')], model='claude-3-haiku-20240307', role='assistant', stop_reason='tool_use', stop_sequence=None, type='message', usage=Usage(cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=407, output_tokens=86, server_tool_use=None, service_tier='standard'))
# Calling ollama ollama:qwen3:30b

# <class 'ollama._types.ChatResponse'>
# model='qwen3:30b' created_at='2025-08-10T00:00:53.979939Z' done=True done_reason='stop' total_duration=3292282583 load_duration=921246333 prompt_eval_count=197 prompt_eval_duration=405703209 eval_count=149 eval_duration=1963442625 message=Message(role='assistant', content="<think>\nOkay, the user is asking for the current weather in Toronto in Celsius. Let me check the tools provided. There's a function called get_weather that requires city and format. The city is Toronto, and the format should be celsius. I need to make sure the parameters are correctly set. The required fields are city and format, both strings. So I'll structure the tool call with those values. Let me double-check the spelling: Toronto is correct, and celsius is the format they want. Alright, that's all I need. Time to format the tool call as specified.\n</think>\n\n", thinking=None, images=None, tool_name=None, tool_calls=[ToolCall(function=Function(name='get_weather', arguments={'city': 'Toronto', 'format': 'celsius'}))])


# Non Tool Call and Non Structured Response

# Calling OpenAI openai:gpt-4o-mini
# <class 'openai.types.responses.response.Response'>
# Response(id='resp_6897f50677cc8193a1806952f7cb3f36053c1fcabf547d96', created_at=1754789126.0, error=None, incomplete_details=None, instructions=None, metadata={}, model='gpt-4o-mini-2024-07-18', object='response', output=[ResponseOutputMessage(id='msg_6897f506f0848193b06a3662b16e6525053c1fcabf547d96', content=[ResponseOutputText(annotations=[], text="Why don't skeletons fight each other? \n\nBecause they don't have the guts!", type='output_text', logprobs=[])], role='assistant', status='completed', type='message')], parallel_tool_calls=True, temperature=1.0, tool_choice='auto', tools=[], top_p=1.0, background=False, max_output_tokens=None, max_tool_calls=None, previous_response_id=None, prompt=None, prompt_cache_key=None, reasoning=Reasoning(effort=None, generate_summary=None, summary=None), safety_identifier=None, service_tier='default', status='completed', text=ResponseTextConfig(format=ResponseFormatText(type='text'), verbosity='medium'), top_logprobs=0, truncation='disabled', usage=ResponseUsage(input_tokens=23, input_tokens_details=InputTokensDetails(cached_tokens=0), output_tokens=17, output_tokens_details=OutputTokensDetails(reasoning_tokens=0), total_tokens=40), user=None, store=True)

# Calling anthropic anthropic:claude-3-haiku-20240307
# <class 'anthropic.types.message.Message'>
# Message(id='msg_01DxWCR1AzMEZSh5B9TWRAKu', content=[TextBlock(citations=None, text="Here's a silly joke for you:\n\nWhy don't scientists trust atoms? Because they make up everything!\n\nHow was that? I hope you got a little chuckle out of it. Let me know if you'd like to hear another joke.", type='text')], model='claude-3-haiku-20240307', role='assistant', stop_reason='end_turn', stop_sequence=None, type='message', usage=Usage(cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=28, output_tokens=55, server_tool_use=None, service_tier='standard'))

# Calling ollama ollama:qwen3:30b
# <class 'ollama._types.ChatResponse'>
# model='qwen3:30b' created_at='2025-08-10T01:25:33.293254Z' done=True done_reason='stop' total_duration=4952211250 load_duration=44045709 prompt_eval_count=25 prompt_eval_duration=160599084 eval_count=362 eval_duration=4746875875 message=Message(role='assistant', content='<think>\nOkay, the user asked for a joke. Let me think of a good one. I should pick something light and appropriate for most people. Maybe a pun or a play on words since those are usually safe.\n\nHmm, I remember a classic one about a chicken crossing the road. But that\'s overused. Let me think of something else. Oh, there\'s a good one about a fish. Wait, no, maybe a different animal. How about a banana? Wait, I think I have one about a duck.\n\nWait, the user might not want a long setup. Short and sweet is better. Let me recall: "Why don\'t scientists trust atoms? Because they make up everything!" Wait, that\'s a good one. But is it too sciencey? Maybe, but it\'s a common pun.\n\nWait, another one: "I told my wife she was drawing her eyebrows too high. She looked surprised." That\'s a pun on "high" and "surprised." But maybe not the best. Let me check if I have a better one.\n\nWait, the classic "Why did the scarecrow win an award? Because he was outstanding in his field!" That\'s a good one. It\'s a pun on "outstanding" and "field." Yeah, that\'s a solid choice. It\'s not offensive, easy to understand, and has a nice twist.\n\nLet me confirm: scarecrow in a field, "outstanding" meaning both excellent and literally standing out. Yep, that works. I\'ll go with that. It\'s a safe, clean joke that should get a chuckle. Alright, I\'ll tell that one.\n</think>\n\nWhy did the scarecrow win an award?  \n*Because he was outstanding in his field!* 🌾', thinking=None, images=None, tool_name=None, tool_calls=None)
