from sqlalchemy import Column, String, Text, JSON, DateTime, ForeignKey, Boolean, Integer
from sqlalchemy.sql import func
from app.core.database import Base

class Report(Base):
    __tablename__ = "reports"

    report_id = Column(String, primary_key=True, index=True) # UUID
    title = Column(String, nullable=False)
    content = Column(JSON, nullable=True) # Structured content (intro, background, etc.)
    status = Column(String, default="generating") # generating, completed, failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_bookmarked = Column(Boolean, default=False)
    
    # Premium / Mentor fields
    report_type = Column(String, default="general", nullable=False) # general, premium
    mentor_comment = Column(Text, nullable=True)
    original_content = Column(JSON, nullable=True) # backup of original content for comparison
    mentor_reviewed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Link to Topic
    topic_id = Column(String, ForeignKey("topics.topic_id"), nullable=False)
    
    # Link to User
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
