import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # App Settings
    DEBUG: bool = False
    SECRET_KEY: str = "your-default-secret-key"
    
    # API Keys
    GEMINI_API_KEY: Optional[str] = ""
    
    # Path Settings
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # CORS Settings
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",
        "https://client-20h.pages.dev"
    ]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
