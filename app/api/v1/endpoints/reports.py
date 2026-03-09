import uuid
import asyncio
from typing import Any, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.config import settings
from app.core.database import AsyncSessionLocal, get_db
from app.models.report import Report
from app.models.topic import Topic
from app.models.user import User
from app.api import deps
from app.schemas.report import (
    BookmarkRequest,
    ReportChatRequest,
    ReportChatResponse,
    ReportGenerateRequest,
    ReportGenerateResponse,
    ReportListResponse,
    ReportResponse,
    ReportUpdateRequest,
)
from app.services import gemini_service
from app.services.report_workflow import run_report_workflow

router = APIRouter()


def serialize_report(report: Report) -> dict:
    content = report.content or {}
    meta = content.get("__meta", {}) if isinstance(content, dict) else {}
    progress = meta.get("progress") if isinstance(meta.get("progress"), int) else None
    phase = meta.get("phase") if isinstance(meta.get("phase"), str) else None
    status_message = meta.get("message") if isinstance(meta.get("message"), str) else None
    return {
        "report_id": report.report_id,
        "status": report.status,
        "title": report.title,
        "content": report.content,
        "created_at": report.created_at,
        "is_bookmarked": report.is_bookmarked,
        "progress": progress,
        "phase": phase,
        "status_message": status_message,
    }


async def generate_report_task(report_id: str, topic_id: str, custom_instructions: str) -> None:
    async with AsyncSessionLocal() as db:
        try:
            async def update_progress(percent: int, phase: str, message: str) -> None:
                progress_result = await db.execute(select(Report).where(Report.report_id == report_id))
                progress_report = progress_result.scalars().first()
                if not progress_report:
                    return
                content = progress_report.content or {}
                meta = content.get("__meta", {})
                meta["progress"] = percent
                meta["phase"] = phase
                meta["message"] = message
                content["__meta"] = meta
                progress_report.content = content
                db.add(progress_report)
                await db.commit()

            topic_result = await db.execute(select(Topic).where(Topic.topic_id == topic_id))
            topic = topic_result.scalars().first()
            if not topic:
                report_result = await db.execute(select(Report).where(Report.report_id == report_id))
                report = report_result.scalars().first()
                if report:
                    report.status = "failed"
                    await db.commit()
                return

            content = await asyncio.wait_for(
                run_report_workflow(
                    subject=topic.subject or "수학",
                    unit_large=topic.unit_large or "미적분",
                    unit_medium=None,
                    unit_small=None,
                    topic_title=topic.title,
                    topic_description=topic.description or "",
                    custom_instructions=custom_instructions or "",
                    on_progress=update_progress,
                ),
                timeout=settings.REPORT_GENERATION_TIMEOUT_SECONDS,
            )

            report_result = await db.execute(select(Report).where(Report.report_id == report_id))
            report = report_result.scalars().first()
            if report:
                final_content = content or {}
                final_content["__meta"] = {
                    "progress": 100,
                    "phase": "completed",
                    "message": "보고서 생성이 완료되었습니다.",
                }
                report.content = final_content
                report.status = "completed"
                db.add(report)
                await db.commit()
        except asyncio.TimeoutError:
            report_result = await db.execute(select(Report).where(Report.report_id == report_id))
            report = report_result.scalars().first()
            if report:
                report.status = "failed"
                report.content = {
                    "error": "보고서 생성 시간이 초과되었습니다.",
                    "hint": "생성 파이프라인이 제한 시간을 넘었습니다. 다시 시도하거나 파이프라인 단계를 줄여 주세요.",
                    "__meta": {
                        "progress": 100,
                        "phase": "failed",
                        "message": "생성 시간이 초과되었습니다.",
                    },
                }
                await db.commit()
        except Exception as e:
            report_result = await db.execute(select(Report).where(Report.report_id == report_id))
            report = report_result.scalars().first()
            if report:
                report.status = "failed"
                report.content = {
                    "error": str(e),
                    "hint": "환경 변수와 LangGraph/LLM API 설정을 확인하세요.",
                }
                await db.commit()


@router.post("/generate", response_model=ReportGenerateResponse)
async def generate_report(
    request: ReportGenerateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    topic_result = await db.execute(select(Topic).where(Topic.topic_id == request.topic_id))
    topic = topic_result.scalars().first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")

    report_id = str(uuid.uuid4())
    report = Report(
        report_id=report_id,
        title=topic.title,
        status="generating",
        content={"__meta": {"progress": 3, "phase": "queued", "message": "생성 작업을 대기열에 등록했습니다."}},
        topic_id=topic.topic_id,
        user_id=current_user.id,
    )
    db.add(report)
    await db.commit()

    background_tasks.add_task(
        generate_report_task,
        report_id,
        request.topic_id,
        request.custom_instructions or "",
    )

    return {
        "report_id": report_id,
        "status": "generating",
        "estimated_time": 30,
    }


@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(
    report_id: str,
    current_user: User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    result = await db.execute(select(Report).where(Report.report_id == report_id))
    report = result.scalars().first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    if report.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    return serialize_report(report)


@router.patch("/{report_id}", response_model=ReportResponse)
async def update_report(
    report_id: str,
    body: ReportUpdateRequest,
    current_user: User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    result = await db.execute(select(Report).where(Report.report_id == report_id))
    report = result.scalars().first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    if report.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    report.content = body.content
    db.add(report)
    await db.commit()
    await db.refresh(report)
    return serialize_report(report)


@router.get("", response_model=List[ReportListResponse])
async def list_reports(
    current_user: User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    query = (
        select(Report, Topic)
        .join(Topic, Report.topic_id == Topic.topic_id)
        .where(Report.user_id == current_user.id)
        .order_by(Report.created_at.desc())
    )
    result = await db.execute(query)
    rows = result.all()

    response = []
    for report, topic in rows:
        response.append(
            ReportListResponse(
                report_id=report.report_id,
                title=report.title,
                created_at=report.created_at,
                status=report.status,
                is_bookmarked=report.is_bookmarked,
                subjects=[topic.subject] if topic and topic.subject else [],
            )
        )
    return response


@router.patch("/{report_id}/bookmark")
async def bookmark_report(
    report_id: str,
    body: BookmarkRequest,
    current_user: User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    result = await db.execute(select(Report).where(Report.report_id == report_id))
    report = result.scalars().first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    if report.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    report.is_bookmarked = body.is_bookmarked
    await db.commit()
    return {"status": "success"}


@router.post("/{report_id}/chat", response_model=ReportChatResponse)
async def chat_with_report(
    report_id: str,
    body: ReportChatRequest,
    current_user: User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    result = await db.execute(select(Report).where(Report.report_id == report_id))
    report = result.scalars().first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    if report.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    reply = await gemini_service.chat_about_report(
        report_title=report.title,
        report_content=report.content or {},
        user_message=body.message,
    )
    return {"reply": reply}


@router.get("/{report_id}/pdf")
async def download_pdf(
    report_id: str,
    current_user: User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    result = await db.execute(select(Report).where(Report.report_id == report_id))
    report = result.scalars().first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    if report.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    raise HTTPException(
        status_code=501,
        detail="PDF export is not implemented on the backend yet. Use the frontend print/export flow instead.",
    )
