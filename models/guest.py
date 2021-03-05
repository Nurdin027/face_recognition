import uuid
from datetime import datetime

from base import Base
from models.sub_device import SubDeviceM
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship


class GuestM(Base):
    __tablename__ = "guest"

    # region Column
    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    name = Column(String)
    filename = Column(String)
    add_time = Column(DateTime, default=datetime.now())

    sub_device_id = Column(String, ForeignKey(SubDeviceM.id))
    sub_device = relationship(SubDeviceM, foreign_keys=sub_device_id)

    # endregion

    def __init__(self,
                 name,
                 filename,
                 sub_device_id,
                 ):
        self.name = name
        self.filename = filename
        self.sub_device_id = sub_device_id

    def json(self):
        return {
            "id": self.id,
            "name": self.name,
            "filename": self.filename,
            "sub_device_id": self.sub_device_id,
            "sub_device": self.sub_device.description,
            "add_time": self.add_time,
        }
