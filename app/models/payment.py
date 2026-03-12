from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.sql import func

from app.core.database import Base


class PaymentOrder(Base):
    __tablename__ = "payment_orders"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    order_id = Column(String(64), unique=True, nullable=False, index=True)
    order_name = Column(String(100), nullable=False)
    package_code = Column(String(50), nullable=False)
    amount = Column(Integer, nullable=False)
    currency = Column(String(10), nullable=False, default="KRW")
    credits_to_add = Column(Integer, nullable=False)
    status = Column(String(30), nullable=False, default="READY", index=True)
    payment_key = Column(String(200), unique=True, nullable=True, index=True)
    method = Column(String(50), nullable=True)
    easy_pay_provider = Column(String(50), nullable=True)
    requested_at = Column(DateTime(timezone=True), server_default=func.now())
    approved_at = Column(DateTime(timezone=True), nullable=True)
    raw_response = Column(JSON, nullable=True)
