from langfuse import Langfuse
from src.config.settings import settings
import logging
import sys

logger = logging.getLogger(__name__)

class LangfuseManager:
    def __init__(self):
        self.client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host
        )

    def get_prompt_and_config(self, prompt_name: str) -> dict:
        """
        FAIL-FAST: Fetches prompt and its JSON configs from Langfuse.
        Raises exception if config keys are missing.
        """
        try:
            prompt_obj = self.client.get_prompt(prompt_name)
            config = prompt_obj.config
            
            if not config or "model" not in config or "temperature" not in config:
                raise KeyError("Missing required keys ('model', 'temperature') in Langfuse prompt config.")
                
            return {
                "system_prompt": prompt_obj.compile(),
                "model_name": config["model"],
                "temperature": config["temperature"]
            }
        except Exception as e:
            logger.critical(f"FATAL: Could not fetch prompt '{prompt_name}' from Langfuse. Error: {e}")
            sys.exit(1)

lf_manager = LangfuseManager()