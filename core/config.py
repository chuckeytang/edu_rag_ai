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
    CHROMA_PATH: str = "./chroma_db"
    INDEX_PATH: str = "./index_store"

    # OSS configuration fields here
    OSS_PROVIDER: str = "ALIYUN"
    OSS_ENDPOINT: str
    OSS_ACCESS_KEY_ID: str
    OSS_ACCESS_KEY_SECRET: str
    OSS_PUBLIC_BUCKET_NAME: str

    # CHROMADB OSS configuration fields here
    OSS_PUBLIC_VECTOR_BUCKET: str
    AWS_S3_ENDPOINT_URL: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_DEFAULT_REGION: str

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
settings = Settings(OSS_ENDPOINT=os.getenv("OSS_ENDPOINT"))
settings = Settings(OSS_ACCESS_KEY_ID=os.getenv("OSS_ACCESS_KEY_ID"))
settings = Settings(OSS_ACCESS_KEY_SECRET=os.getenv("OSS_ACCESS_KEY_SECRET"))
settings = Settings(OSS_PUBLIC_BUCKET_NAME=os.getenv("OSS_PUBLIC_BUCKET_NAME"))

settings = Settings(OSS_PUBLIC_VECTOR_BUCKET=os.getenv("OSS_PUBLIC_VECTOR_BUCKET"))
settings = Settings(AWS_S3_ENDPOINT_URL=os.getenv("AWS_S3_ENDPOINT_URL"))
settings = Settings(AWS_ACCESS_KEY_ID=os.getenv("AWS_ACCESS_KEY_ID"))
settings = Settings(AWS_SECRET_ACCESS_KEY=os.getenv("AWS_SECRET_ACCESS_KEY"))
settings = Settings(AWS_DEFAULT_REGION=os.getenv("AWS_DEFAULT_REGION"))