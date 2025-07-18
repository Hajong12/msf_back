# core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    gemini_api_key: str
    debug: bool = False

    class Config:
        env_file = ".env"  # .env 파일에서 읽어옴

settings = Settings()