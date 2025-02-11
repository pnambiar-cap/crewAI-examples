from typing import List
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

#add for capgemini generative engine interface for getting LLM object
import os
import logging
import yaml
import litellm
from generative_engine_litellm.generative_engine_handler import GenerativeEngineLLM


config_path = os.path.join(os.getcwd(), 'generative_engine_config.yaml')
print(config_path)
# Load the custom LLM handler with the config path
generative_engine_llm = GenerativeEngineLLM(config_path=config_path)

# Register the custom handler with LiteLLM
litellm.custom_provider_map = [
	{"provider": "generative-engine", "custom_handler": generative_engine_llm}
]
print(f"Custom provider map: {litellm.custom_provider_map}")

#=========================================
# Configure logging to capture debug information
# Explicitly configure the root logger to capture all logging levels and direct to both console and file
logging_level = logging.DEBUG
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Remove any existing handlers to avoid duplicates
if logger.hasHandlers():
    logger.handlers.clear()

# Add handlers to both stream (console) and file
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging_level)
file_handler = logging.FileHandler("/tmp/crewai_debug_forced.log")
file_handler.setLevel(logging_level)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

# Also set logging for third-party libraries explicitly
litellm_logger = logging.getLogger("litellm")
litellm_logger.setLevel(logging_level)
litellm_logger.addHandler(stream_handler)
litellm_logger.addHandler(file_handler)

crewai_logger = logging.getLogger("crewai")
crewai_logger.setLevel(logging_level)
crewai_logger.addHandler(stream_handler)
crewai_logger.addHandler(file_handler)

generative_engine_litellm = logging.getLogger("generative_engine_litellm")
generative_engine_litellm.setLevel(logging_level)
generative_engine_litellm.addHandler(stream_handler)
generative_engine_litellm.addHandler(file_handler)

# Configure logging to capture debug information for this script
logger.info("Logging is configured and started.")

#============================================

#Choose a model to use for the LLM
#model_name = 'generative-engine/anthropic.claude-v2'
#model_name = 'generative-engine/openai.gpt-4o'
#model_name = 'generative-engine/openai.o1-mini' 
#model_name = 'generative-engine/openai.o1-preview'
#model_name = 'generative-engine/openai.gpt-3.5-turbo'
#model_name = 'generative-engine/openai.gpt-4'

# Uncomment the following line to use an example of a custom tool
# from marketing_posts.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from pydantic import BaseModel, Field

class MarketStrategy(BaseModel):
	"""Market strategy model"""
	name: str = Field(..., description="Name of the market strategy")
	tatics: List[str] = Field(..., description="List of tactics to be used in the market strategy")
	channels: List[str] = Field(..., description="List of channels to be used in the market strategy")
	KPIs: List[str] = Field(..., description="List of KPIs to be used in the market strategy")

class CampaignIdea(BaseModel):
	"""Campaign idea model"""
	name: str = Field(..., description="Name of the campaign idea")
	description: str = Field(..., description="Description of the campaign idea")
	audience: str = Field(..., description="Audience of the campaign idea")
	channel: str = Field(..., description="Channel of the campaign idea")

class Copy(BaseModel):
	"""Copy model"""
	title: str = Field(..., description="Title of the copy")
	body: str = Field(..., description="Body of the copy")

@CrewBase
class MarketingPostsCrew():
	"""MarketingPosts crew"""

	


	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def lead_market_analyst(self) -> Agent:
		model_name = 'generative-engine/openai.gpt-4o'
		return Agent(
			config=self.agents_config['lead_market_analyst'],
			llm=LLM(model=model_name, timeout=180, max_tokens=32768),
			tools=[SerperDevTool(), ScrapeWebsiteTool()],
			verbose=True,
			memory=False,
		)

	@agent
	def chief_marketing_strategist(self) -> Agent:
		model_name = 'generative-engine/openai.gpt-4o' 
		return Agent(
			config=self.agents_config['chief_marketing_strategist'],
			llm=LLM(model=model_name, timeout=180, max_tokens=32768),
			tools=[SerperDevTool(), ScrapeWebsiteTool()],
			verbose=True,
			memory=False,
		)

	@agent
	def creative_content_creator(self) -> Agent:
		model_name = 'generative-engine/openai.gpt-4o'
		return Agent(
			config=self.agents_config['creative_content_creator'],
			llm=LLM(model=model_name, timeout=180, max_tokens=32768),
			verbose=True,
			memory=False,
		)

	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
			agent=self.lead_market_analyst()
		)

	@task
	def project_understanding_task(self) -> Task:
		return Task(
			config=self.tasks_config['project_understanding_task'],
			agent=self.chief_marketing_strategist()
		)

	@task
	def marketing_strategy_task(self) -> Task:
		return Task(
			config=self.tasks_config['marketing_strategy_task'],
			agent=self.chief_marketing_strategist(),
			output_json=MarketStrategy
		)

	@task
	def campaign_idea_task(self) -> Task:
		return Task(
			config=self.tasks_config['campaign_idea_task'],
			agent=self.creative_content_creator(),
   		output_json=CampaignIdea
		)

	@task
	def copy_creation_task(self) -> Task:
		return Task(
			config=self.tasks_config['copy_creation_task'],
			agent=self.creative_content_creator(),
   		context=[self.marketing_strategy_task(), self.campaign_idea_task()],
			output_json=Copy
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the MarketingPosts crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
