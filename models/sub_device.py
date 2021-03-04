import uuid

from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from base import Base
from models.main_device import MainDeviceM


class SubDeviceM(Base):
    __tablename__ = 'sub_device'
    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    channel = Column(Integer)
    description = Column(String, comment="Deskripsi atau nama kamera")
    detect_stat = Column(Integer, default=1)
    add_by = Column(String)
    main_device_id = Column(String, ForeignKey('main_device.id'))
    main_device = relationship(MainDeviceM, backref="sub_device")

    def json(self):
        return {
            "id": self.id,
            "channel": self.channel,
            "description": self.description,
            "main_device_id": self.main_device_id,
            "host": self.main_device.host,
            "detect_stat": self.detect_stat,
            "add_by": self.add_by,
        }
