FROM python:3.10-slim
WORKDIR /app

# 设置清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 复制requirements文件(从项目根目录)
COPY ../requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制整个项目(从项目根目录)
COPY .. .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]