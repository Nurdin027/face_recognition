import uuid

from sqlalchemy import Column, String, Integer

from base import Base


class CustomerGroupM(Base):
    __tablename__ = "customer_group"
    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    name = Column(String)
    address = Column(String)
    email = Column(String)
    telephone = Column(String)
    status = Column(Integer)
    add_by = Column(String)
