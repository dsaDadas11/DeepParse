#!/bin/bash
set -e

echo "=== application startup ==="

echo "Waiting for database connection..."
python - <<'PY'
import os
import time
import sys
import psycopg2
from psycopg2 import OperationalError

max_retries = 30
for retry_count in range(1, max_retries + 1):
    try:
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
        conn.close()
        print('Database connection established.')
        break
    except OperationalError:
        print(f'Database not ready yet ({retry_count}/{max_retries})')
        time.sleep(2)
else:
    print('Database connection failed after retries.')
    sys.exit(1)
PY

echo "Running database migration..."
python - <<'PY'
import os
from alembic.config import Config
from alembic import command
from sqlalchemy import create_engine, text

try:
    engine = create_engine(os.environ['DATABASE_URL'])
    alembic_cfg = Config('alembic.ini')
    with engine.connect() as conn:
        version_table_exists = conn.execute(
            text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'alembic_version')")
        ).scalar()
        legacy_tables_exist = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name IN ('sessions', 'messages', 'knowledgebases')
                """
            )
        ).scalar() >= 3
        if not version_table_exists and legacy_tables_exist:
            print('Detected legacy schema, stamping baseline revision...')
            command.stamp(alembic_cfg, '980b32f130df')

    command.upgrade(alembic_cfg, 'head')
    print('Database migration completed.')
except Exception as exc:
    print(f'Migration failed: {exc}')
    raise
PY

echo "Starting application service..."
exec "$@"
