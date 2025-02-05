
import asyncio
import dotenv
import os
import pytz
from datetime import datetime
from autogen_agentchat.agents import AssistantAgent, SocietyOfMindAgent
from autogen_agentchat.teams import (
    MagenticOneGroupChat,
)
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_agentchat.teams._group_chat._magentic_one._magentic_one_orchestrator import MagenticOneOrchestrator
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

dotenv.load_dotenv()

# Create the token provider
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

az_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
    model=os.getenv("AZURE_OPENAI_COMPLETION_MODEL"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_ad_token_provider=token_provider,  # Optional if you choose key-based authentication.
    # api_key="sk-...", # For key-based authentication.
)

o1_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment="o1-mini",
    model="o1-mini",
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_ad_token_provider=token_provider,  # Optional if you choose key-based authentication.
    # api_key="sk-...", # For key-based authentication.
)

reasoning_agent = AssistantAgent(name="reasoning_agent", model_client=o1_model_client, 
                            system_message=None,
                            description="A helpful assistant that can check the quality of the conversation and provide feedback to the agents. The checker agent can provide feedback on the quality of the conversation, the relevance of the responses, and the overall satisfaction of the user. The checker agent can also provide suggestions for improvement to the agents and should be consulted before completing the last task.",
                            tools=None)

# Define a tool
async def check_conversation(messages: str) -> str:
    print("executing check_conversation")
    response = await reasoning_agent.on_messages(
                    [TextMessage(content=f"Check the following messages for inconsistencies, open questions and contradictions and give concrete feedback. Messages: {messages}", source="user")], CancellationToken()
                )
    print(response)
    return f"This is my feedback {response}."

# Define a tool
async def get_weather(city: str) -> str:
    print("executing get_weather")
    return f"The weather in {city} is 73 degrees and Sunny."

async def get_medical_history(username: str) -> str:
    "Get the medical history for a given username with known allergies and food restrictions."
    print("executing get_medical_history")
    return f"{username} has an allergy to peanuts and eggs."

async def get_available_incredients(location: str) -> str:
    "Get the available incredients for a given location."
    print("executing get_available_incredients")
    return f"Available incredients in {location} are: eggs, milk, bread, peanuts, beer, wine, salmon, spinache, oil and butter."

def get_current_username(input: str) -> str:
    "Get the username of the current user."
    print("executing get_current_username")
    return "Dennis"

def get_current_location_of_user(username: str) -> str:
    "Get the current timezone location of the user for a given username."
    print("executing get_current_location")
    print(username)
    if "Dennis" in username:
        return "Europe/Berlin"
    else:
        return "America/New_York"

def get_current_time(location: str) -> str:
    "Get the current time in the given location. The pytz is used to get the timezone for that location. Location names should be in a format like America/Seattle, Asia/Bangkok, Europe/London. Anything in Germany should be Europe/Berlin"
    try:
        print("get current time for location: ", location)
        timezone = pytz.timezone(location)
        # Get the current time in the timezone
        now = datetime.now(timezone)
        current_time = now.strftime("%I:%M:%S %p")
        return current_time
    except Exception as e:
        print("Error: ", e)
        return "Sorry, I couldn't find the timezone for that location."
    
users_agent = AssistantAgent(
    "users_agent",
    model_client=az_model_client,
    tools=[get_current_username, get_medical_history],
    description="A helpful assistant that can knows things about the user like the username.",
    system_message="You are a helpful assistant that can retrieve the username of the current user.",
)

location_agent = AssistantAgent(
    "location_agent",
    model_client=az_model_client,
    tools=[get_current_location_of_user],
    description="A assistant that can find the physical location of a user.",
    system_message="You are a helpful assistant that can suggest details for a location and can utilize any context information provided.",
)

time_agent = AssistantAgent(
    "time_agent",
    model_client=az_model_client,
    tools=[get_current_time],
    description="A helpful assistant that knows time in a specific location.",
    system_message="You are a helpful assistant that can retrieve the current time for a given location.",
)

chef_agent = AssistantAgent(
    "chef_agent",
    model_client=az_model_client,
    tools=[get_available_incredients],
    description="A helpful assistant that can suggest meals and dishes for the right time of the day, location, available ingredients, user preferences and allergies.",
    system_message="You are a helpful assistant that can recommend dishes for the right time of the day, location, available ingredients and user preferences. Make sure you ask for individual food preferences and allergies as input.",
)

summary_agent = AssistantAgent(
    "summary_agent",
    model_client=az_model_client,
    description="A helpful assistant that can summarize details about conversations.",
    system_message="You are a helpful assistant that can take in all of the suggestions and advice from the other agents and leverage them to answer questions. You must ensure that you use that the other agents can solve the problem. When all open questions have been answered, you can respond with TERMINATE.",
)

consultation_agent = AssistantAgent(name="consultation_agent", model_client=az_model_client, 
                            system_message="Your task is to check the complete message flow for inconsistencies, open questions and contradictions and give concrete feedback. You should also provide suggestions for improvement to the agents and should be consulted before completing the last task.",
                            description="A helpful assistant that can check the quality of the conversation and provide feedback to the agents. The checker agent can provide feedback on the quality of the conversation, the relevance of the responses, and the overall satisfaction of the user. The checker agent can also provide suggestions for improvement to the agents and should be consulted before completing the last task.",
                            tools=[check_conversation])


async def main() -> None:

    inner_termination = MaxMessageTermination(20)
    magenticteam = MagenticOneGroupChat([users_agent, location_agent, time_agent, chef_agent, consultation_agent], model_client=az_model_client, termination_condition=inner_termination)

    # Run the team and stream messages to the console
    stream = magenticteam.run_stream(task="I want to have something to eat. What would you recommend?.")
    await Console(stream)

    # async for message in stream:
    #     print(message)

asyncio.run(main())
