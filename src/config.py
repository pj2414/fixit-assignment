# Configuration for Fixit GenAI Backend

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Settings
    app_name: str = "Fixit GenAI Backend"
    app_version: str = "1.0.0"
    debug: bool = False
    
   
