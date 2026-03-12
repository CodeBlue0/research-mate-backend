from fastapi import APIRouter
from app.api.v1.endpoints import auth, curriculum, payments, topics, reports

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(curriculum.router, prefix="/curriculum", tags=["curriculum"])
api_router.include_router(topics.router, prefix="/topics", tags=["topics"])
api_router.include_router(reports.router, prefix="/reports", tags=["reports"])
api_router.include_router(payments.router, prefix="/payments", tags=["payments"])
