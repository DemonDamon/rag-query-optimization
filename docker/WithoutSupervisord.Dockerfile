# 使用一个基础的Docker镜像，可以根据你的需求选择合适的镜像
FROM bitnami/python:3.11.7


# 设置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装必要的软件包
# 清空原有的sources.list文件
RUN echo "" > /etc/apt/sources.list

# 添加新的镜像源
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-backports main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb https://security.debian.org/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list

# 更新apt-get包管理器
RUN apt-get update && \
    apt-get install -y build-essential wget git vim net-tools curl unzip zip ntp apt-file && \
    apt-get install -y automake autoconf libssl-dev libbz2-dev libexpat1-dev libffi-dev \
    libgdbm-dev libreadline-dev zlib1g-dev libgl1 libglib2.0-0 supervisor && \
    apt-get install -y libmagic1 libmagic-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 设置pip主要源和备用源(切换为国内源，如不是在国内请忽略)
RUN pip config set global.index-url http://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.extra-index-url https://pypi.org/simple/


# 复制requirements.txt文件到Docker镜像中
COPY requirements.txt .

# 安装Python依赖包
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 直接修改符号链接 将sh默认设为bash
RUN ln -sf /bin/bash /bin/sh

