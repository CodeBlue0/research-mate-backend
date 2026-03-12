from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from sqlalchemy import text, update
from app.core.config import settings
from app.core.database import Base, engine, close_connectors, AsyncSessionLocal
from app.models.credit_transaction import CreditTransaction
from app.models.payment import PaymentOrder
from app.models.report import Report
from app.services import gemini_service
from app import models  # noqa: F401


async def _fail_stale_generating_reports() -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=settings.STALE_REPORT_TIMEOUT_MINUTES)
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(Report)
            .where(Report.status == "generating")
            .where(Report.created_at < cutoff)
            .values(
                status="failed",
                content={
                    "error": "Server restarted before report generation completed.",
                    "hint": "보고서를 다시 생성해 주세요.",
                    "__meta": {
                        "progress": 100,
                        "phase": "failed",
                        "message": "서버 재시작 또는 워커 중단으로 생성 작업이 종료되었습니다.",
                    },
                },
            )
        )
        await session.commit()


async def _ensure_payment_columns() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS credit_balance INTEGER NOT NULL DEFAULT 0"))
        await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS customer_key VARCHAR(50)"))
        await conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_customer_key ON users (customer_key)"))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS payment_orders (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                order_id VARCHAR(64) NOT NULL,
                order_name VARCHAR(100) NOT NULL,
                package_code VARCHAR(50) NOT NULL,
                amount INTEGER NOT NULL,
                currency VARCHAR(10) NOT NULL DEFAULT 'KRW',
                credits_to_add INTEGER NOT NULL,
                status VARCHAR(30) NOT NULL DEFAULT 'READY',
                payment_key VARCHAR(200),
                method VARCHAR(50),
                easy_pay_provider VARCHAR(50),
                requested_at TIMESTAMPTZ DEFAULT NOW(),
                approved_at TIMESTAMPTZ,
                raw_response JSON
            )
        """))
        await conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_payment_orders_order_id ON payment_orders (order_id)"))
        await conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_payment_orders_payment_key ON payment_orders (payment_key) WHERE payment_key IS NOT NULL"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_payment_orders_user_id ON payment_orders (user_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_payment_orders_status ON payment_orders (status)"))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS credit_transactions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                package_code VARCHAR(50) NOT NULL,
                delta INTEGER NOT NULL,
                transaction_type VARCHAR(20) NOT NULL,
                reason VARCHAR(100) NOT NULL,
                payment_order_id INTEGER REFERENCES payment_orders(id),
                report_id VARCHAR(36),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_credit_transactions_user_id ON credit_transactions (user_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_credit_transactions_package_code ON credit_transactions (package_code)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_credit_transactions_transaction_type ON credit_transactions (transaction_type)"))
        await conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_credit_transactions_payment_order_id_unique ON credit_transactions (payment_order_id) WHERE payment_order_id IS NOT NULL"))
        await conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_credit_transactions_report_id_unique ON credit_transactions (report_id) WHERE report_id IS NOT NULL"))


async def _backfill_credit_transactions() -> None:
    async with AsyncSessionLocal() as session:
        done_orders = (
            await session.execute(
                text("""
                    SELECT po.id, po.user_id, po.package_code, po.credits_to_add
                    FROM payment_orders po
                    WHERE po.status = 'DONE'
                      AND NOT EXISTS (
                        SELECT 1
                        FROM credit_transactions ct
                        WHERE ct.payment_order_id = po.id
                      )
                """)
            )
        ).all()

        for order_id, user_id, package_code, credits_to_add in done_orders:
            session.add(
                CreditTransaction(
                    user_id=user_id,
                    package_code=package_code,
                    delta=credits_to_add,
                    transaction_type="EARN",
                    reason="payment_completed",
                    payment_order_id=order_id,
                )
            )

        if done_orders:
            await session.commit()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.ENVIRONMENT == "production":
        if settings.SECRET_KEY == "CHANGE_THIS_TO_A_SECURE_SECRET_KEY":
            raise RuntimeError("SECRET_KEY must be set in production")
        if "*" in settings.cors_allow_origins_list:
            raise RuntimeError("CORS_ALLOW_ORIGINS must not contain '*' in production")
    if settings.AUTO_CREATE_TABLES:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    await _ensure_payment_columns()
    await _backfill_credit_transactions()
    await _fail_stale_generating_reports()
    yield
    await engine.dispose()
    await close_connectors()


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts_list)

from app.api.v1.api import api_router

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "environment": settings.ENVIRONMENT,
        "llm": gemini_service.provider_status(),
        "openai_model": settings.OPENAI_MODEL,
        "openai_base": settings.OPENAI_API_BASE,
    }
