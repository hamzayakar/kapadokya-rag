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
        If it fails, it raises an exception and stops the application.
        """
        try:
            prompt_obj = self.client.get_prompt(prompt_name)
            
            # Extract config, provide sensible fail-safe defaults for config ONLY, not the prompt
            config = prompt_obj.config if prompt_obj.config else {}
            
            return {
                "system_prompt": prompt_obj.compile(),
                "model_name": config.get("model", "gemini-1.5-flash"), # Default to cheap model just in case
                "temperature": config.get("temperature", 0.0)
            }
        except Exception as e:
            logger.critical(f"FATAL: Could not fetch prompt '{prompt_name}' from Langfuse. System halting. Error: {e}")
            sys.exit(1) # Fail-fast: Stop execution immediately

lf_manager = LangfuseManager()