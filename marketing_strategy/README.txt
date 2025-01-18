The original README.md does not indicate the package dependencies were switched to uv from poetry.
To use uv for package management do the following:
    pip install uv
    uv init
    uv sync
    uv run python marketing_posts

For porting this to use Capgemini Generative Engine we need to use the custom LiteLLM interface to Generative Engine. 
We will not be using the OPENAI API KEY instead use the generative_engine_config.yaml file
For more details see repo - https://github.com/pnambiar-cap/generative_engine_litellm