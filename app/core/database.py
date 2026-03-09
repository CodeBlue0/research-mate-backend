import asyncio
from typing import Dict

from google.cloud.sql.connector import Connector, IPTypes
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.core.config import settings

Base = declarative_base()

# Event-loop specific connector instances to avoid ConnectorLoopError.
_connectors_by_loop: Dict[int, Connector] = {}

async def get_connector():
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    connector = _connectors_by_loop.get(loop_id)
    if connector is None:
        connector = Connector(loop=loop, refresh_strategy="LAZY")
        _connectors_by_loop[loop_id] = connector
    return connector

async def getconn():
    connector = await get_connector()
    conn = await connector.connect_async(
        settings.INSTANCE_CONNECTION_NAME,
        "asyncpg",
        user=settings.DB_USER,
        password=settings.DB_PASS,
        db=settings.DB_NAME,
        ip_type=IPTypes.PUBLIC, # Or PRIVATE if running within VPC
    )
    return conn


async def close_connectors():
    for loop_id, connector in list(_connectors_by_loop.items()):
        try:
            await connector.close_async()
        except Exception:
            try:
                connector.close()
            except Exception:
                pass
        _connectors_by_loop.pop(loop_id, None)

# Create the engine with the async_creator
# Note: We can't easily create the engine globally with an async creator if the connector needs async init.
# But the Connector object itself doesn't need async init, only the connect_async method.
# Wait, Connector() is not async. 

# However, creating engine usually happens at module level or startup.
# Let's create a lazy engine wrapper or just initialize it.

# If we are running locally without cloud sql connector setup variables, this might fail immediately.
# We'll assume the variables are present or will be.

if settings.DATABASE_URL:
    engine = create_async_engine(settings.DATABASE_URL)
elif settings.INSTANCE_CONNECTION_NAME and (settings.ENVIRONMENT == "production" or settings.USE_CLOUD_SQL_IN_DEV):
    engine = create_async_engine(
        "postgresql+asyncpg://",
        async_creator=getconn,
    )
else:
    raise RuntimeError(
        "Database configuration is missing. Set DATABASE_URL or configure Cloud SQL "
        "with INSTANCE_CONNECTION_NAME, DB_USER, DB_PASS, and DB_NAME."
    )

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
