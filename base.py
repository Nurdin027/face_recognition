from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine('postgresql://teda:m0n0w4ll@192.168.5.8:5432/tims')
Session = sessionmaker(bind=engine)

Base = declarative_base()