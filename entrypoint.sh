#!/bin/bash
set -e

echo "Checking if database needs seeding..."

# Run the check in a subshell so set -e doesn't kill the script on exit 1
NEEDS_SEED=0
python - <<'EOF' || NEEDS_SEED=1
import psycopg2, os, sys
url = os.environ.get("DATABASE_URL", "")
try:
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public' AND table_name='products'")
    exists = cur.fetchone()[0]
    conn.close()
    if not exists:
        print("Tables not found — seeding required.")
        sys.exit(1)
    else:
        print("Tables already exist, skipping seed.")
        sys.exit(0)
except Exception as e:
    print(f"DB check failed: {e}")
    sys.exit(1)
EOF

if [ "$NEEDS_SEED" -ne 0 ]; then
    echo "Seeding database..."
    python data/seed.py
fi

exec streamlit run ui/app.py --server.port=8501 --server.address=0.0.0.0
