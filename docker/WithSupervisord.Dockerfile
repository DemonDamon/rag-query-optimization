FROM b66ypcgaicisp01-f8a759e8.ecis.guangzhou-2.cmecloud.cn/prod/query-rewrite_base:1.0.0

WORKDIR /

COPY . /app/query-rewrite/

RUN chmod +x /app/query-rewrite/docker/docker-entrypoint.sh


ENTRYPOINT ["sh", "/app/query-rewrite/docker/docker-entrypoint.sh"]
