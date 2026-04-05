import asyncio
import os
import sys

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from app.core.config import settings

async def main():
    engine = create_async_engine(settings.DATABASE_URL)
    async with engine.begin() as conn:
        print("Adding report_type column...")
        await conn.execute(text("ALTER TABLE reports ADD COLUMN IF NOT EXISTS report_type VARCHAR NOT NULL DEFAULT 'general'"))
        
        print("Adding mentor_comment column...")
        await conn.execute(text("ALTER TABLE reports ADD COLUMN IF NOT EXISTS mentor_comment TEXT"))
        
        print("Adding original_content column...")
        await conn.execute(text("ALTER TABLE reports ADD COLUMN IF NOT EXISTS original_content JSON"))
        
        print("Adding mentor_reviewed_at column...")
        await conn.execute(text("ALTER TABLE reports ADD COLUMN IF NOT EXISTS mentor_reviewed_at TIMESTAMP WITH TIME ZONE"))
        
        print("Migration complete!")

if __name__ == "__main__":
    asyncio.run(main())
