import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from base import Base
from models.customer_group import CustomerGroupM


class UserM(Base):
    __tablename__ = 'users'

    id = Column(String, primary_key=True, nullable=False, unique=True, default=lambda: uuid.uuid4().hex)
    name = Column(String, nullable=False)
    email = Column(String)
    telephone = Column(String)
    nik_nrp = Column(String, unique=True)
    add_by = Column(String)
    add_time = Column(DateTime, nullable=False, default=lambda: datetime.now())

    customer_group_id = Column(String, ForeignKey(CustomerGroupM.id))
    customer_group = relationship(CustomerGroupM, foreign_keys=customer_group_id)
