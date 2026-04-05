from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.core.database import get_db
from app.models.credit_transaction import CreditTransaction
from app.models.payment import PaymentOrder
from app.models.report import Report
from app.models.user import User
from datetime import datetime

router = APIRouter()

ADMIN_EMAIL = "coldbootcp@gmail.com"


class AdminCreditAdjustRequest(BaseModel):
    package_code: str
    delta: int  # positive = add, negative = remove


class AdminUserPackageCredit(BaseModel):
    package_code: str
    credit_balance: int


class AdminUserResponse(BaseModel):
    id: int
    email: str
    name: str | None
    credit_balance: int
    created_at: str
    package_credits: list[AdminUserPackageCredit]


class AdminCreditAdjustResponse(BaseModel):
    user_id: int
    package_code: str
    new_balance: int
    delta: int


class AdminReportResponse(BaseModel):
    report_id: str
    title: str
    user_email: str
    created_at: datetime
    status: str


class AdminReportReviewRequest(BaseModel):
    content: dict[str, Any]
    mentor_comment: str


async def _require_admin(current_user: User = Depends(deps.get_current_user)) -> User:
    if current_user.email != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


async def _get_package_credit_balances(db: AsyncSession, user_id: int) -> dict[str, int]:
    result = await db.execute(
        select(
            CreditTransaction.package_code,
            func.coalesce(func.sum(CreditTransaction.delta), 0),
        )
        .where(CreditTransaction.user_id == user_id)
        .group_by(CreditTransaction.package_code)
    )

    return {
        package_code: max(0, int(total or 0))
        for package_code, total in result.all()
    }


@router.get("/users", response_model=list[AdminUserResponse])
async def list_users(
    _admin: User = Depends(_require_admin),
    db: AsyncSession = Depends(get_db),
) -> Any:
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    users = result.scalars().all()

    response = []
    for user in users:
        pcb = await _get_package_credit_balances(db, user.id)
        package_credits = [
            AdminUserPackageCredit(package_code=code, credit_balance=balance)
            for code, balance in pcb.items()
        ]
        # Always include both package codes even if zero
        existing_codes = {p.package_code for p in package_credits}
        for code in ("basic", "premium-review"):
            if code not in existing_codes:
                package_credits.append(AdminUserPackageCredit(package_code=code, credit_balance=0))

        response.append(
            AdminUserResponse(
                id=user.id,
                email=user.email,
                name=user.name,
                credit_balance=user.credit_balance or 0,
                created_at=user.created_at.isoformat() if user.created_at else "",
                package_credits=package_credits,
            )
        )
    return response


@router.post("/users/{user_id}/credits", response_model=AdminCreditAdjustResponse)
async def adjust_user_credits(
    user_id: int,
    body: AdminCreditAdjustRequest,
    _admin: User = Depends(_require_admin),
    db: AsyncSession = Depends(get_db),
) -> Any:
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    tx_type = "EARN" if body.delta > 0 else "SPEND"
    reason = "admin_adjustment"

    db.add(
        CreditTransaction(
            user_id=user.id,
            package_code=body.package_code,
            delta=body.delta,
            transaction_type=tx_type,
            reason=reason,
        )
    )
    user.credit_balance = max(0, (user.credit_balance or 0) + body.delta)
    await db.commit()
    await db.refresh(user)

    pcb = await _get_package_credit_balances(db, user.id)
    new_balance = pcb.get(body.package_code, 0)

    return AdminCreditAdjustResponse(
        user_id=user.id,
        package_code=body.package_code,
        new_balance=new_balance,
        delta=body.delta,
    )


@router.get("/reports/awaiting-review", response_model=list[AdminReportResponse])
async def list_awaiting_reviews(
    _admin: User = Depends(_require_admin),
    db: AsyncSession = Depends(get_db),
) -> Any:
    result = await db.execute(
        select(Report, User.email)
        .join(User, Report.user_id == User.id)
        .where(Report.status == "awaiting_review")
        .order_by(Report.created_at.asc())
    )
    rows = result.all()
    return [
        AdminReportResponse(
            report_id=row.Report.report_id,
            title=row.Report.title,
            user_email=row.email,
            created_at=row.Report.created_at,
            status=row.Report.status,
        )
        for row in rows
    ]


@router.post("/reports/{report_id}/review")
async def review_report(
    report_id: str,
    body: AdminReportReviewRequest,
    _admin: User = Depends(_require_admin),
    db: AsyncSession = Depends(get_db),
) -> Any:
    result = await db.execute(select(Report).where(Report.report_id == report_id))
    report = result.scalars().first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    # Backup original content if not already backed up
    if report.original_content is None:
        report.original_content = report.content

    report.content = body.content
    report.mentor_comment = body.mentor_comment
    report.mentor_reviewed_at = func.now()
    report.status = "completed"

    db.add(report)
    await db.commit()
    return {"status": "success"}
