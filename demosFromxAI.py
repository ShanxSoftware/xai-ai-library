import os
import base64
import json
from xai_sdk import Client
from xai_sdk.chat import Response, user, system, assistant, image, tool, tool_result
from xai_sdk.search import SearchParameters
from typing import Dict, Callable, Any, List, Optional, Literal
from pydantic import BaseModel, Field, model_validator
client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    timeout=3600, # Override default timeout with longer timeout for reasoning models
)

def reasoned_chat():
    chat = client.chat.create(
        model="grok-4",
        messages=[system("You are a highly intelligent AI assistant.")],
    )

    chat.append(user("What is 101*3?"))
    response = chat.sample()

    print("Reasoning Content:")
    print(response.reasoning_content)

    print("Final Response:")
    print(response.content)

    print("Number of completeion tokens: ")
    print(response.usage.completion_tokens)

    print("Number of reasoning tokens:")
    print(response.usage.reasoning_tokens)

def search_chat(): # Need to create sources and filters (from and to dates, max results), RSS Sources could include emergency, news, and weather services.
    chat = client.chat.create(
        model="grok-4",
        search_parameters=SearchParameters(mode="auto"),
        return_citations=True,
    )

    chat.append(user("Provide mea  digest of world news of the week before July 9, 2025"))
    response = chat.sample()
    print(response.content)
    print(response.citations)


"""
Source: https://docs.x.ai/docs/guides/image-understanding
"""
def image_understanding(image_path = ""):
    chat = client.chat.create(
        model="grok-4"
    )
    base64_image = encode_image(image_path)
    chat.append(
        user(
            "What is in this image?", 
            image(image_url=f"data:image/jpeg;base64,{base64_image}", detail="high"),
        )
    )

    response = chat.sample()
    print(response.content)
    print(response.usage.prompt_image_tokens)

"""
Source: https://docs.x.ai/docs/guides/streaming-response
"""
def stream_chat(): 
    chat = client.chat.create(model="grok-4")
    chat.append(
        system("You ar eGrok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."),
    )
    chat.append(
        user("What is the meaning of life, the universe, and everything?")
    )

    for response, chunk in chat.stream():
        print(chunk.content, end="", flush=True) # Each chunk's content
        print(response.content, end="", flush=True) # the response object auto-accumulates the chunks

    print(response.content) # The full response

"""
Source: https://docs.x.ai/docs/guides/function-calling
"""

class TemperatureRequest(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    unit: Literal["celsius", "fahrenheit"] = Field(
        "fahrenheit", description="Temperature unit"
    )

class CeilingRequest(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")

def get_current_temperature(request: TemperatureRequest):
    temperature = 59 if request.unit.lower() == "fahrenheit" else 15
    return {
        "location": request.location,
        "temperature": temperature,
        "unit": request.unit,
    }

def get_current_ceiling(request: CeilingRequest):
    return {
        "location": request.location,
        "ceiling": 15000,
        "ceiling_type": "broken",
        "unit": "ft",
    }

# Generate the JSON schema from the Pydantic models

get_current_temperature_schema = TemperatureRequest.model_json_schema()
get_current_ceiling_schema = CeilingRequest.model_json_schema()

# Definition of parameters with Pydantic JSON schema

tool_definitions = [
    tool(
        name="get_current_temperature",
        description="Get the current temperature in a given location",
        parameters=get_current_temperature_schema,
    ),
    tool(
        name="get_current_ceiling",
        description="Get the current cloud ceiling in a given location",
        parameters=get_current_ceiling_schema,
    ),
]
tools_map = {
    "get_current_temperature": get_current_temperature, 
    "get_current_ceiling": get_current_ceiling,
}

def call_functions(): 
    chat = client.chat.create(
        model="grok-4", 
        tools=tool_definitions,
        tool_choice="auto",
    )
    chat.append(user("What's the temperature like in San Francisco?"))
    response = chat.sample()
    # You can inspect the response tool calls which contains a tool call
    print(response.tool_calls)
    # Append assistant message including tool calls to messages
    chat.append(response)

    # check if there is any tool calls in response body, wrap this in a function to clean up code
    if response.tool_calls: 
        for tool_call in response.tool_calls: 
            # Get the tool function name and arguments Grok wants to call
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            # Call one of the tool function defined earlier with arguments
            result = tools_map[function_name](**function_args)

            # Append the result from tool function call to the chat message history
            chat.append(tool_result(result))

    # Send function results back to Grok
    response = chat.sample()
    print(response.content)

def encode_image(image_path): 
    """
    Source: https://docs.x.ai/docs/guides/image-understanding
    Convert image file to string for sending to grok.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string