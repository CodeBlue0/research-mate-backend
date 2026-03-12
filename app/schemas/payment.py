from datetime import datetime
from typing import Any

from pydantic import BaseModel


class PaymentPackage(BaseModel):
    code: str
    name: str
    description: str
    credits: int
    credit_balance: int = 0
    amount: int
    original_amount: int
    badge: str
    claim_limit: int
    claim_count: int = 0
    claim_remaining: int = 0


class PaymentSummaryResponse(BaseModel):
    customer_key: str
    credit_balance: int
    usage_count: int
    packages: list[PaymentPackage]


class PaymentOrderCreateRequest(BaseModel):
    package_code: str


class PaymentPromotionClaimRequest(BaseModel):
    package_code: str


class PaymentOrderCreateResponse(BaseModel):
    order_id: str
    order_name: str
    amount: int
    customer_key: str
    customer_email: str
    customer_name: str


class PaymentConfirmRequest(BaseModel):
    paymentKey: str
    orderId: str
    amount: int


class PaymentConfirmResponse(BaseModel):
    order_id: str
    amount: int
    credit_balance: int
    credits_added: int
    method: str | None = None
    easy_pay_provider: str | None = None
    approved_at: datetime | None = None
    raw: dict[str, Any]


class PaymentPromotionClaimResponse(BaseModel):
    order_id: str
    amount: int
    credit_balance: int
    credits_added: int
    package_code: str
    package_credit_balance: int
    package_claim_count: int
    package_claim_remaining: int
