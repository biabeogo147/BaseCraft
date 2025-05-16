FROM redis/redis-stack:latest

ENV REDIS_ARGS="--requirepass PASSWORD"

EXPOSE 6379 8001

CMD ["sh", "-c", "redis-server $REDIS_ARGS"]