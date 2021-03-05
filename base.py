from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine('postgresql://teda:m0n0w4ll@localhost:5432/tims')

Base = declarative_base()
