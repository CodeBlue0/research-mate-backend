from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class ReportGenerateRequest(BaseModel):
    topic_id: str
    report_type: str = "general"
    custom_instructions: Optional[str] = None

class ReportGenerateResponse(BaseModel):
    report_id: str
    status: str
    estimated_time: int
    charged_package_code: Optional[str] = None
    remaining_credit_balance: Optional[int] = None

class ReportResponse(BaseModel):
    report_id: str
    status: str
    title: str
    content: Optional[Dict[str, Any]] = None
    created_at: datetime
    is_bookmarked: bool
    progress: Optional[int] = None
    phase: Optional[str] = None
    status_message: Optional[str] = None

    report_type: str
    mentor_comment: Optional[str] = None
    original_content: Optional[Dict[str, Any]] = None
    mentor_reviewed_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class ReportListResponse(BaseModel):
    report_id: str
    title: str
    subjects: Optional[List[str]] = None
    created_at: datetime
    status: str
    is_bookmarked: bool

    class Config:
        from_attributes = True

class BookmarkRequest(BaseModel):
    is_bookmarked: bool


class ReportChatRequest(BaseModel):
    message: str


class ReportChatResponse(BaseModel):
    reply: str


class ReportUpdateRequest(BaseModel):
    content: Dict[str, Any]
    title: Optional[str] = None
