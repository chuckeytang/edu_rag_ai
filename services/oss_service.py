# In services/oss_service.py
import os
import io
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from fastapi import HTTPException
from dotenv import load_dotenv
import tempfile
import logging
from core.config import settings

logger = logging.getLogger(__name__)

class OssService:
    def __init__(self):
        self.endpoint_url = settings.OSS_ENDPOINT
        self.access_key_id = settings.OSS_ACCESS_KEY_ID
        self.access_key_secret = settings.OSS_ACCESS_KEY_SECRET
        self.public_bucket_name = settings.OSS_PUBLIC_BUCKET_NAME
        self.private_bucket_name = settings.OSS_PRIVATE_BUCKET_NAME
        self.region_name = settings.AWS_DEFAULT_REGION

        if not all([self.endpoint_url, self.access_key_id, self.access_key_secret]):
            raise ValueError("OSS configuration is missing in environment variables.")

        # --- ADDED LOGGING FOR DEBUGGING ---
        logger.info(f"Initializing OssService with the following configuration:")
        logger.info(f"  Endpoint URL: {self.endpoint_url}")
        logger.info(f"  Access Key ID: {'*' * (len(self.access_key_id) - 4) + self.access_key_id[-4:] if self.access_key_id else 'Not Set'}")

        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.access_key_secret,
            region_name=self.region_name,
            config=Config(
                s3={'addressing_style': 'virtual'}, # Use virtual-hosted style, which is standard for accelerate endpoints
                connect_timeout=10, # Add a connection timeout
                read_timeout=10
        )
        )
        logging.info(f"OssService initialized")

    def download_file_to_temp(self, object_key: str, bucket_name: str) -> str:
        """
        Downloads a file from a SPECIFIED OSS bucket to a new local temporary directory.
        
        Args:
            object_key (str): The key of the object to download.
            bucket_name (str): The name of the bucket to download from.

        Returns:
            str: The full local file path of the downloaded file.
        """
        try:
            # 创建一个唯一的临时目录来存放下载的文件
            temp_dir = tempfile.mkdtemp(prefix="oss_downloads_")
            # 从object_key中提取原始文件名
            original_filename = os.path.basename(object_key)
            # 构造完整的本地文件路径
            local_file_path = os.path.join(temp_dir, original_filename)

            logger.info(f"Downloading s3://{bucket_name}/{object_key} to temporary file '{local_file_path}'...")
            
            # 使用 boto3 的 download_file 方法
            self.s3_client.download_file(bucket_name, object_key, local_file_path)
            
            logger.info("Download to temporary file complete.")
            return local_file_path
            
        except ClientError as e:
            logger.error(f"Botocore ClientError downloading from bucket '{bucket_name}' for key '{object_key}': {e.response['Error']['Code']}", exc_info=True)
            if e.response['Error']['Code'] == '404' or e.response['Error']['Code'] == 'NoSuchKey':
                raise HTTPException(status_code=404, detail=f"File not found in OSS: {object_key}")
            else:
                raise HTTPException(status_code=500, detail=f"OSS download error: {e}")
        except Exception as e:
            logger.error(f"Generic exception downloading '{object_key}': {str(e)}", exc_info=True)
            raise HTTPException(status_code=503, detail=f"Could not connect to OSS endpoint or another error occurred. Original error: {str(e)}")
