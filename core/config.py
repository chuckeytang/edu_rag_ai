import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
load_dotenv()
class Settings(BaseSettings):
    PROJECT_NAME: str = "LLM Ninja API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DASHSCOPE_API_KEY: str
    DATA_DIR: str = "data/pdf"
    DATA_CONFIG_DIR: str = "data/.config"
    BACKEND_CORS_ORIGINS: list[str] = ["*"]
    class Config:
        case_sensitive = True
        env_file = ".env"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__init_data_dir()
    def __init_data_dir(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.DATA_CONFIG_DIR, exist_ok=True)
settings = Settings(DASHSCOPE_API_KEY=os.getenv("DASHSCOPE_API_KEY"))