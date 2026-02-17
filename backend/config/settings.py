from pydantic import BaseSettings


class Settings(BaseSettings):
    GOOGLE_API_KEY: str

    class Config:
        env_file = ".env"


settings = Settings()