import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from base import Base
from models.sub_device import SubDeviceM


class LogUnknownM(Base):
    __tablename__ = 'log_unknown'

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    add_time = Column(DateTime, default=datetime.now())
    photo = Column(String)

    sub_device_id = Column(String, ForeignKey('sub_device.id'))
    sub_device = relationship(SubDeviceM, foreign_keys=sub_device_id)

    def __init__(self, photo, sub_device_id):
        self.photo = photo
        self.sub_device_id = sub_device_id

    def json(self):
        return {
            "id": self.id,
            "photo": self.photo,
            "sub_device_id": self.sub_device_id,
            "sub_device_name": self.sub_device.desc,
            "main_device_name": self.sub_device.main_device.name,
            "add_time": str(self.add_time),
        }
