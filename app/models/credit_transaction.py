from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.sql import func

from app.core.database import Base


class CreditTransaction(Base):
    __tablename__ = "credit_transactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    package_code = Column(String(50), nullable=False, index=True)
    delta = Column(Integer, nullable=False)
    transaction_type = Column(String(20), nullable=False, index=True)
    reason = Column(String(100), nullable=False)
    payment_order_id = Column(Integer, ForeignKey("payment_orders.id"), nullable=True, index=True)
    report_id = Column(String(36), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
