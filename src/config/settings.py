from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # LLM
    gemini_api_key: str = Field(default=..., env="GEMINI_API_KEY")
    
    # Qdrant
    qdrant_url: str = Field(default=..., env="QDRANT_URL")
    qdrant_api_key: str = Field(default=..., env="QDRANT_API_KEY")
    
    # Langfuse
    langfuse_public_key: str = Field(default=..., env="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default=..., env="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", env="LANGFUSE_HOST")

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore" 
    )

# Singleton pattern: import settings from anywhere in the app
settings = Settings()