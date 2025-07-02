import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

print("="*50)
print(f"[DEBUG] Before  {os.environ.get('AWS_DEFAULT_REGION')}")
print(f"[DEBUG] Before  {os.environ.get('OSS_PRIVATE_BUCKET_NAME')}")
print(f"[DEBUG] Before  {os.environ.get('AWS_S3_ENDPOINT_URL')}")

load_dotenv()

print(f"[DEBUG] After  {os.environ.get('AWS_DEFAULT_REGION')}")
print(f"[DEBUG] After  {os.environ.get('OSS_PRIVATE_BUCKET_NAME')}")
print(f"[DEBUG] After  {os.environ.get('AWS_S3_ENDPOINT_URL')}")
print("="*50)

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
    OSS_PRIVATE_BUCKET_NAME: str

    # CHROMADB OSS configuration fields here
    OSS_PUBLIC_VECTOR_BUCKET: str
    AWS_S3_ENDPOINT_URL: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_DEFAULT_REGION: str

    DEEPSEEK_API_BASE: str
    DEEPSEEK_API_KEY: str

    # 使用 SettingsConfigDict 来配置 Pydantic
    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        extra="ignore" # 如果 .env 中有模型中未定义的字段，会忽略而不是报错
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__init_data_dir()

    def __init_data_dir(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.DATA_CONFIG_DIR, exist_ok=True)

settings = Settings()
