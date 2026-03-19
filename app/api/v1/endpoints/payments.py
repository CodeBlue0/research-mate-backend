import asyncio
import base64
import json
import secrets
import string
from datetime import datetime
from urllib import error, request
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.core.database import get_db
from app.core.config import settings
from app.models.credit_transaction import CreditTransaction
from app.models.payment import PaymentOrder
from app.models.user import User
from app.schemas.payment import (
    PaymentConfirmRequest,
    PaymentConfirmResponse,
    PaymentOrderCreateRequest,
    PaymentOrderCreateResponse,
    PaymentPackage,
    PaymentPromotionClaimRequest,
    PaymentPromotionClaimResponse,
    PaymentSummaryResponse,
)

router = APIRouter()

PAYMENT_PACKAGES = {
    "basic": PaymentPackage(
        code="basic",
        name="기본 요금제",
        description="심화 탐구 보고서 생성 3회",
        credits=3,
        amount=0,
        original_amount=29000,
        badge="입문용",
        claim_limit=3,
    ),
    "premium-review": PaymentPackage(
        code="premium-review",
        name="프리미엄 검수 요금제",
        description="프리미엄 검수 포함 보고서 생성 3회",
        credits=3,
        amount=0,
        original_amount=99000,
        badge="추천",
        claim_limit=1,
    ),
}


def _generate_order_id(length: int = 20) -> str:
    alphabet = string.ascii_letters + string.digits + "-_"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def _get_or_create_customer_key(user: User) -> str:
    if user.customer_key:
        return user.customer_key
    user.customer_key = str(uuid4())
    return user.customer_key


def _parse_approved_at(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


async def _get_promotion_claim_count(db: AsyncSession, user_id: int, package_code: str | None = None) -> int:
    conditions = [
        PaymentOrder.user_id == user_id,
        PaymentOrder.method == "PROMOTION",
        PaymentOrder.status == "DONE",
    ]
    if package_code:
        conditions.append(PaymentOrder.package_code == package_code)

    result = await db.execute(
        select(func.count(PaymentOrder.id)).where(*conditions)
    )
    return int(result.scalar() or 0)


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


async def _get_usage_count(db: AsyncSession, user_id: int) -> int:
    result = await db.execute(
        select(func.count(CreditTransaction.id)).where(
            CreditTransaction.user_id == user_id,
            CreditTransaction.transaction_type == "SPEND",
        )
    )
    return int(result.scalar() or 0)


def _confirm_with_toss(payment_key: str, order_id: str, amount: int) -> dict:
    if not settings.TOSS_PAYMENTS_SECRET_KEY:
        raise HTTPException(status_code=500, detail="TOSS_PAYMENTS_SECRET_KEY is not configured")

    auth = base64.b64encode(f"{settings.TOSS_PAYMENTS_SECRET_KEY}:".encode("utf-8")).decode("utf-8")
    payload = json.dumps({
        "paymentKey": payment_key,
        "orderId": order_id,
        "amount": amount,
    }).encode("utf-8")

    req = request.Request(
        "https://api.tosspayments.com/v1/payments/confirm",
        data=payload,
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        detail = "Payment confirmation failed"
        try:
            parsed = json.loads(body)
            detail = parsed.get("message") or parsed.get("code") or detail
        except json.JSONDecodeError:
            detail = body or detail
        raise HTTPException(status_code=400, detail=detail)
    except error.URLError:
        raise HTTPException(status_code=502, detail="Failed to reach TossPayments")


@router.get("/summary", response_model=PaymentSummaryResponse)
async def get_payment_summary(
    current_user: User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(get_db),
):
    customer_key = _get_or_create_customer_key(current_user)
    package_credit_balances = await _get_package_credit_balances(db, current_user.id)
    usage_count = await _get_usage_count(db, current_user.id)
    packages: list[PaymentPackage] = []
    for package in PAYMENT_PACKAGES.values():
        claim_count = await _get_promotion_claim_count(db, current_user.id, package.code)
        packages.append(
            PaymentPackage(
                **package.model_dump(exclude={"credit_balance", "claim_count", "claim_remaining"}),
                credit_balance=package_credit_balances.get(package.code, 0),
                claim_count=claim_count,
                claim_remaining=max(0, package.claim_limit - claim_count),
            )
        )
    await db.commit()
    await db.refresh(current_user)
    return PaymentSummaryResponse(
        customer_key=customer_key,
        credit_balance=current_user.credit_balance or 0,
        usage_count=usage_count,
        packages=packages,
    )


@router.post("/orders", response_model=PaymentOrderCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_payment_order(
    body: PaymentOrderCreateRequest,
    current_user: User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(get_db),
):
    package = PAYMENT_PACKAGES.get(body.package_code)
    if not package:
        raise HTTPException(status_code=404, detail="Unknown package")

    customer_key = _get_or_create_customer_key(current_user)
    order = PaymentOrder(
        user_id=current_user.id,
        order_id=_generate_order_id(),
        order_name=f"{package.name} {package.credits}회 이용권",
        package_code=package.code,
        amount=package.amount,
        currency="KRW",
        credits_to_add=package.credits,
        status="READY",
    )
    db.add(order)
    await db.commit()
    await db.refresh(order)
    await db.refresh(current_user)

    return PaymentOrderCreateResponse(
        order_id=order.order_id,
        order_name=order.order_name,
        amount=order.amount,
        customer_key=customer_key,
        customer_email=current_user.email,
        customer_name=current_user.name or current_user.email,
    )


@router.post("/promotions/claim", response_model=PaymentPromotionClaimResponse, status_code=status.HTTP_201_CREATED)
async def claim_promotion_package(
    body: PaymentPromotionClaimRequest,
    current_user: User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(get_db),
):
    package = PAYMENT_PACKAGES.get(body.package_code)
    if not package:
        raise HTTPException(status_code=404, detail="Unknown package")

    claim_count = await _get_promotion_claim_count(db, current_user.id, package.code)
    if claim_count >= package.claim_limit:
        raise HTTPException(
            status_code=400,
            detail=f"{package.name}은(는) 계정당 최대 {package.claim_limit}회까지 받을 수 있습니다.",
        )

    order = PaymentOrder(
        user_id=current_user.id,
        order_id=_generate_order_id(),
        order_name=f"{package.name} 이벤트 이용권",
        package_code=package.code,
        amount=package.amount,
        currency="KRW",
        credits_to_add=package.credits,
        status="DONE",
        method="PROMOTION",
        raw_response={
            "type": "promotion",
            "package_code": package.code,
            "credits_added": package.credits,
        },
        approved_at=datetime.utcnow(),
    )
    db.add(order)
    await db.flush()
    db.add(
        CreditTransaction(
            user_id=current_user.id,
            package_code=package.code,
            delta=package.credits,
            transaction_type="EARN",
            reason="promotion_claimed",
            payment_order_id=order.id,
        )
    )
    current_user.credit_balance = (current_user.credit_balance or 0) + package.credits

    await db.commit()
    await db.refresh(current_user)
    await db.refresh(order)
    package_credit_balances = await _get_package_credit_balances(db, current_user.id)

    return PaymentPromotionClaimResponse(
        order_id=order.order_id,
        amount=order.amount,
        credit_balance=current_user.credit_balance,
        credits_added=order.credits_to_add,
        package_code=order.package_code,
        package_credit_balance=package_credit_balances.get(package.code, 0),
        package_claim_count=claim_count + 1,
        package_claim_remaining=max(0, package.claim_limit - (claim_count + 1)),
    )


@router.post("/confirm", response_model=PaymentConfirmResponse)
async def confirm_payment(
    body: PaymentConfirmRequest,
    current_user: User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(PaymentOrder).where(PaymentOrder.order_id == body.orderId, PaymentOrder.user_id == current_user.id)
    )
    order = result.scalars().first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    if order.amount != body.amount:
        raise HTTPException(status_code=400, detail="Amount mismatch")

    if order.status == "DONE":
        return PaymentConfirmResponse(
            order_id=order.order_id,
            amount=order.amount,
            credit_balance=current_user.credit_balance or 0,
            credits_added=0,
            method=order.method,
            easy_pay_provider=order.easy_pay_provider,
            approved_at=order.approved_at,
            raw=order.raw_response or {},
        )

    payment = await asyncio.to_thread(_confirm_with_toss, body.paymentKey, body.orderId, body.amount)
    if payment.get("status") != "DONE":
        raise HTTPException(status_code=400, detail="Payment is not completed")

    order.status = payment.get("status", "DONE")
    order.payment_key = payment.get("paymentKey", body.paymentKey)
    order.method = payment.get("method")
    order.easy_pay_provider = (payment.get("easyPay") or {}).get("provider")
    order.approved_at = _parse_approved_at(payment.get("approvedAt"))
    order.raw_response = payment

    current_user.credit_balance = (current_user.credit_balance or 0) + order.credits_to_add
    db.add(
        CreditTransaction(
            user_id=current_user.id,
            package_code=order.package_code,
            delta=order.credits_to_add,
            transaction_type="EARN",
            reason="payment_confirmed",
            payment_order_id=order.id,
        )
    )

    await db.commit()
    await db.refresh(current_user)

    return PaymentConfirmResponse(
        order_id=order.order_id,
        amount=order.amount,
        credit_balance=current_user.credit_balance,
        credits_added=order.credits_to_add,
        method=order.method,
        easy_pay_provider=order.easy_pay_provider,
        approved_at=order.approved_at,
        raw=payment,
    )
