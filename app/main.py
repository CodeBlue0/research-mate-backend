from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from app.core.config import settings
from app.core.database import Base, engine, close_connectors
from app.services import gemini_service
from app import models  # noqa: F401


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
