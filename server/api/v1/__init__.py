"""
API v1 Router
"""
from fastapi import APIRouter
from server.api.v1 import health

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])

# More routers will be added here as we build endpoints
# api_router.include_router(inference.router, prefix="/inference", tags=["inference"])
# api_router.include_router(avatar.router, prefix="/avatar", tags=["avatar"])

