import os

import redis
from rq import Queue, Worker, Connection

redis_url = os.getenv("REDISTOGO_URL", "redis://localhost:6379")

conn = redis.from_url(redis_url)

q_runner = Queue("q_runner", connection=conn)

if __name__ == "__main__":
    with Connection(conn):
        worker = Worker(q_runner, connection=conn)
        worker.work()
