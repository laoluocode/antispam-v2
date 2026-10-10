FROM python:3.12-slim

# 时区数据（日志与判定时间需要）
RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Shanghai
RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone

WORKDIR /app

# 依赖单独一层，代码变更不触发重装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 与 compose 的 volume 挂载点一致：数据库放外面，重建镜像不丢数据
VOLUME ["/app/data"]
ENV DB_NAME=/app/data/antispam.db

CMD ["python", "main.py"]
