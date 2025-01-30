from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    openai_api_key: str = Field(...)
    hf_token: str = Field(...)

    class Config:
        env_file = '.env'
        extra = "allow"

settings = Settings()