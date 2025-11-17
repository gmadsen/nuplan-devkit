#!/usr/bin/env python3
"""
ABOUTME: Verify that database connection pooling is working correctly.
ABOUTME: Counts connection creations and demonstrates pooling behavior.
"""

import sqlite3
import tempfile
from pathlib import Path

import sqlalchemy

from nuplan.database.common.db import SessionManager


def main():
    """Demonstrate connection pooling reduces connection creation."""
    # Create temporary test database
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db_path = Path(temp_db.name)

    # Setup test table
    conn = sqlite3.connect(str(temp_db_path))
    conn.execute('CREATE TABLE test (id INTEGER, value TEXT)')
    for i in range(100):
        conn.execute('INSERT INTO test VALUES (?, ?)', (i, f'value_{i}'))
    conn.commit()
    conn.close()

    print("Testing database connection pooling...")
    print("=" * 60)

    # Track connection creations
    connection_count = 0

    def counting_creator():
        nonlocal connection_count
        connection_count += 1
        print(f"  Creating connection #{connection_count}")
        return sqlite3.connect(str(temp_db_path), check_same_thread=False)

    # Create SessionManager with pooling
    session_manager = SessionManager(
        counting_creator,
        pool_size=3,  # Small pool for demonstration
        max_overflow=2
    )

    print("\n1. Initializing SessionManager (connections created lazily)...")
    print(f"   Connections created: {connection_count}")

    print("\n2. Getting session (creates engine but no connections yet)...")
    session = session_manager.session
    print(f"   Connections created: {connection_count}")

    print("\n3. Executing 10 queries (should reuse pooled connections)...")
    for i in range(10):
        result = session.execute(sqlalchemy.text("SELECT * FROM test WHERE id = :id"), {"id": i})
        data = result.fetchall()
        if i == 0:
            print(f"   Query {i+1}: fetched {len(data)} rows")

    print(f"   Total connections created: {connection_count}")
    print(f"   Expected: ≤ {session_manager._pool_size + session_manager._max_overflow} (pool_size + max_overflow)")

    # Verify pooling is working
    if connection_count <= session_manager._pool_size:
        print("\n✓ SUCCESS: Connection pooling is working!")
        print(f"  Only {connection_count} connection(s) created for 10 queries")
        print("  Connections are being reused from the pool")
    else:
        print("\n✗ WARNING: More connections created than expected")
        print(f"  Created {connection_count}, expected ≤ {session_manager._pool_size}")

    # Cleanup
    temp_db_path.unlink(missing_ok=True)

    print("=" * 60)


if __name__ == '__main__':
    main()
