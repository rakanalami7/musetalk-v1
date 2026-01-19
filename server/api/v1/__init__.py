"""
API v1 Router
"""
from fastapi import APIRouter
from server.api.v1 import health, session, generate

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(session.router, prefix="/session", tags=["session"])
api_router.include_router(generate.router, prefix="/generate", tags=["generate"])

