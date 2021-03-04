import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from base import Base
from models.sub_device import SubDeviceM


class GuestM(Base):
    __tablename__ = "guest"

    # region Column
    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    name = Column(String)
    filename = Column(String)
    add_time = Column(DateTime, default=datetime.now())

    sub_division_id = Column(String, ForeignKey(SubDeviceM.id))
    sub_division = relationship(SubDeviceM, foreign_keys=sub_division_id)

    # endregion

    def __init__(self,
                 name,
                 filename,
                 sub_division_id,
                 ):
        self.name = name
        self.filename = filename
        self.sub_division_id = sub_division_id

    def json(self):
        return {
            "id": self.id,
            "name": self.name,
            "filename": self.filename,
            "sub_division_id": self.sub_division_id,
            "sub_division": self.sub_division.name,
            "add_time": self.add_time,
        }
