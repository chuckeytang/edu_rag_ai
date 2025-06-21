# In services/oss_service.py
import os
import boto3
from botocore.exceptions import ClientError
from fastapi import HTTPException
from dotenv import load_dotenv
import tempfile
import logging

load_dotenv()

class OssService:
    def __init__(self):
        self.endpoint_url = os.getenv("OSS_ENDPOINT")
        self.access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
        self.access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
        self.bucket_name = os.getenv("OSS_BUCKET_NAME", "cheese-dev-public") # Default to public

        if not all([self.endpoint_url, self.access_key_id, self.access_key_secret]):
            raise ValueError("OSS configuration is missing in environment variables.")

        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.access_key_secret
        )
        logging.info(f"OssService initialized for bucket: {self.bucket_name}")

    def download_file(self, object_key: str) -> str:
        """Downloads a file from OSS to a local temporary directory and returns the path."""
        try:
            original_filename = os.path.basename(object_key)
            temp_dir = tempfile.mkdtemp(prefix="oss_downloads_")
            local_file_path = os.path.join(temp_dir, original_filename)

            logging.info(f"Downloading s3://{self.bucket_name}/{object_key} to {local_file_path}...")
            self.s3_client.download_file(self.bucket_name, object_key, local_file_path)
            logging.info("Download complete.")
            return local_file_path
        except ClientError as e:
            logging.error(f"Failed to download file from OSS: {e}")
            if e.response['Error']['Code'] == '404':
                raise HTTPException(status_code=404, detail=f"File not found in OSS: {object_key}")
            else:
                raise HTTPException(status_code=500, detail=f"OSS download error: {e}")

oss_service = OssService()