import asyncio
import os
import sys

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text
from app.core.database import engine, Base
# Import models to ensure they are registered with Base.metadata
from app.models.user import User
from app.models.report import Report
from app.models.topic import Topic
from app.models.payment import PaymentOrder
from app.models.credit_transaction import CreditTransaction

USER_TABLES = [
    "credit_transactions",
    "payment_orders",
    "reports",
    "topics",
    "users",
]

async def main():
    print(f"Connecting to database to reset user tables: {USER_TABLES}")
    async with engine.begin() as conn:
        # Disable foreign key checks if necessary, or drop in correct order
        # PostgreSQL doesn't have a simple global 'set foreign_key_checks=0' like MySQL, 
        # so we drop with CASCADE or in order.
        for table in USER_TABLES:
            print(f"Dropping table {table} if exists...")
            await conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
        
        print("Recreating user tables...")
        # Since we only want to create specific tables from Base.metadata, 
        # we can't easily use create_all() if it includes curriculum tables.
        # But we can create them manually or filtered.
        
        # Another way: creation order matters.
        # We'll just use Base.metadata.create_all but we want to avoid recreating existing ones or 
        # creating curriculum if they were missing (they shouldn't be).
        
        # Actually, create_all() only creates tables that don't exist. 
        # So it's perfect after we dropped the user tables!
        await conn.run_sync(Base.metadata.create_all)
        
        print("User tables reset successfully (curriculum tables preserved).")

if __name__ == "__main__":
    asyncio.run(main())
