import uuid

from sqlalchemy import Column, Integer, String, Float

from base import Base


class MainDeviceM(Base):
    __tablename__ = 'main_device'

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    name = Column(String)
    customer_group_id = Column(String)
    host = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    status = Column(Integer, default=1)
    add_by = Column(String)

    def json(self):
        return {
            "id": self.id,
            "name": self.name,
            "customer_group_id": self.customer_group_id,
            "host": self.host,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "status": self.status,
            "add_by": self.add_by,
        }
