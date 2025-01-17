
import asyncio
import time
from typing import Optional
import dotenv
import os
from dataclasses import dataclass
from datetime import datetime
from autogen_agentchat.agents import AssistantAgent, SocietyOfMindAgent
from autogen_agentchat.teams import (
    MagenticOneGroupChat,
)
from autogen_agentchat.teams._group_chat._magentic_one._magentic_one_orchestrator import MagenticOneOrchestrator
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from IPython.display import clear_output

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


class Robot:
    position_x : int = 0
    position_y : int = 0
    battery : int = 5
    name : str = ""

    def __init__(self, name: str, position_x: int, position_y: int, battery: int = 5):
        self.name = name
        self.battery = battery
        self.position_x = position_x
        self.position_y = position_y

class Wall:
    position_x : int = 0
    position_y : int = 0

    def __init__(self, position_x: int, position_y: int):
        self.position_x = position_x
        self.position_y = position_y

@dataclass
class BoardEvent:
    robot_name: str
    direction: str
    new_position_x: int
    new_position_y: int
    message: str
    success: bool

class Board:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.robots = []
        self.walls = []
        self.last_event = None

    def is_on_board(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def __str__(self):
        return f"Board(width={self.width}, height={self.height})"
    
    width = 0
    height = 0
    robots = []
    walls = []
    last_event: BoardEvent = Optional[BoardEvent]
    
    def add_robot(self, robot: Robot):
        self.robots.append(robot)

    def add_wall(self, wall: Wall):
        self.walls.append(wall)

    def move_robot(self, robot_name: str, direction: str) -> BoardEvent: 
        robot: Robot = next( r for r in self.robots if r.name == robot_name)
        if robot.battery <= 0:
            self.last_event = BoardEvent(robot_name, direction, robot.position_x, robot.position_y, f"{robot_name} has no battery left.", False)
            return self.last_event
        
        if (direction == "north" and robot.position_y == 0) or \
            (direction == "south" and robot.position_y == self.height - 1) or \
            (direction == "east" and robot.position_x == 0) or \
            (direction == "west" and robot.position_x == self.width - 1):
            self.last_event = BoardEvent(robot_name, direction, robot.position_x, robot.position_y, f"{robot_name} cannot move {direction}.", False)
            return self.last_event
        
        new_position_x = robot.position_x
        new_position_y = robot.position_y
        if direction == "south":
            new_position_y += 1
        elif direction == "north":
            new_position_y -= 1
        elif direction == "west":
            new_position_x -= 1
        elif direction == "east":
            new_position_x += 1

        for wall in self.walls:
            if wall.position_x == new_position_x and wall.position_y == new_position_y:
                self.last_event = BoardEvent(robot_name, direction, robot.position_x, robot.position_y, f"{robot_name} cannot move {direction}. There is a wall in the way.", False)
                return self.last_event

        for other_robot in self.robots:
            if other_robot.position_x == new_position_x and other_robot.position_y == new_position_y:
                self.last_event = BoardEvent(robot_name, direction, robot.position_x, robot.position_y, f"{robot_name} cannot move {direction}. {other_robot.name} is in the way.", False)
                return self.last_event
        
        robot.position_x = new_position_x
        robot.position_y = new_position_y
        robot.battery -= 1
        self.last_event = BoardEvent(robot_name, direction, new_position_x, new_position_y, f"{robot_name} moved {direction}.", True)

        return self.last_event

    def print_board(self):
        clear_console()
        for y in range(self.height):
            for x in range(self.width):
                field_emtpy = True
                for wall in self.walls:
                    if wall.position_x == x and wall.position_y == y:
                        print("#", end=" ")
                        field_emtpy = False
                        break
                for robot in self.robots:
                    if robot.position_x == x and robot.position_y == y:
                        print(robot.name, end=" ")
                        field_emtpy = False
                        break
                if field_emtpy:
                    print("-", end=" ")
            print()
        if self.last_event:
            print(self.last_event.message)
        for robot in self.robots:
            print("Robot", robot.name, "is at", robot.position_x, robot.position_y, "with", robot.battery, "battery left.")
            print(self.what_can_robot_see(robot.name))

    def what_can_robot_see(self, robot_name: str):
        robot: Robot = next( r for r in self.robots if r.name == robot_name)

        robot_observation = []
        if (robot.position_x == 0):
            robot_observation.append("I can see the end of the board in the west.")
        
        if (robot.position_x == self.width - 1):
            robot_observation.append("I can see the end of the board in the east.")

        if (robot.position_y == 0):
            robot_observation.append("I can see the end of the board in the north.")

        if (robot.position_y == self.height - 1):
            robot_observation.append("I can see the end of the board in the south.")

        for wall in self.walls:
            if wall.position_x == robot.position_x and wall.position_y < robot.position_y:
                robot_observation.append("I can see a wall to the north. I cannot see what is behind it.")
            if wall.position_x == robot.position_x and wall.position_y > robot.position_y:
                robot_observation.append("I can see a wall to the south. I cannot see what is behind it.")
            if wall.position_y == robot.position_y and wall.position_x < robot.position_x:
                robot_observation.append("I can see a wall to the west. I cannot see what is behind it.")
            if wall.position_y == robot.position_y and wall.position_x > robot.position_x:
                robot_observation.append("I can see a wall to the east. I cannot see what is behind it.")
        
        for other_robot in self.robots:
            if other_robot.name != robot_name:
                if other_robot.position_x == robot.position_x and other_robot.position_y < robot.position_y:
                    robot_observation.append(f"I can see {other_robot.name} to the north.")
                if other_robot.position_x == robot.position_x and other_robot.position_y > robot.position_y:
                    robot_observation.append(f"I can see {other_robot.name} to the south.")
                if other_robot.position_y == robot.position_y and other_robot.position_x < robot.position_x:
                    robot_observation.append(f"I can see {other_robot.name} to the west.")
                if other_robot.position_y == robot.position_y and other_robot.position_x > robot.position_x:
                    robot_observation.append(f"I can see {other_robot.name} to the east.")
        
        return " ".join(robot_observation)

def clear_console():
    # For Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # For macOS and Linux
    else:
        _ = os.system('clear')

async def main() -> None:
    board = Board(20, 20)
    robot1 = Robot("B", 1, 1)
    robot2 = Robot("G", 4, 4)
    board.add_wall(Wall(2, 2))
    board.add_wall(Wall(3, 2))
    board.add_wall(Wall(4, 2))
    board.add_robot(robot1)
    board.add_robot(robot2)   
    board.print_board()
    board.move_robot("B", "south")
    while board.last_event.success:
        time.sleep(1)
        board.print_board()
        board.move_robot("G", "east")
        board.print_board()

asyncio.run(main())
