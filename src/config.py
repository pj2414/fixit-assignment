

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    
    app_name: str = "Fixit GenAI Backend"
    app_version: str = "1.0.0"
    debug: bool = False
    
    
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    llm_timeout: int = 60
    llm_max_retries: int = 3
    
   
    hot_threshold: float = 0.7
    warm_threshold: float = 0.4
    
    
    good_call_threshold: float = 0.6
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
