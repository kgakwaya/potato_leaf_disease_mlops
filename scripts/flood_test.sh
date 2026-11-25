#!/usr/bin/env bash
# Run the Locust flood test against 1, 2 and 3 API containers and summarise
# latency / throughput. Requires Docker + `pip install locust`.
#   bash scripts/flood_test.sh            # defaults: 100 users, 60 s per run
#   USERS=200 DURATION=90s bash scripts/flood_test.sh
set -euo pipefail
USERS=${USERS:-100}
SPAWN=${SPAWN:-10}
DURATION=${DURATION:-60s}
CONTAINERS=${CONTAINERS:-"1 2 3"}
HOST=${HOST:-http://localhost:8080}
mkdir -p locust_results

for n in $CONTAINERS; do
  echo "=== $n container(s), $USERS users, $DURATION ==="
  docker compose up -d --build --scale api="$n"
  docker compose restart nginx >/dev/null
  # wait until every replica answers
  for i in $(seq 60); do curl -sf "$HOST/health" >/dev/null && break; sleep 2; done
  sleep 5
  locust -f locustfile.py --host "$HOST" --headless -u "$USERS" -r "$SPAWN" -t "$DURATION" \
         --csv "locust_results/c${n}_u${USERS}" --html "locust_results/c${n}_u${USERS}.html" --only-summary || true
done

"${PYTHON:-python3}" - <<'PY'
import csv, glob, re
rows = []
for f in sorted(glob.glob("locust_results/c*_u*_stats.csv")):
    n, u = re.findall(r"c(\d+)_u(\d+)", f)[0]
    for r in csv.DictReader(open(f)):
        if r["Name"] == "/predict":
            rows.append((int(n), int(u), r["Request Count"], r["Failure Count"], float(r["Average Response Time"]),
                         r["50%"], r["95%"], r["99%"], float(r["Requests/s"])))
print("| Containers | Users | Requests | Failures | Avg (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Req/s |")
print("|---|---|---|---|---|---|---|---|---|")
for r in sorted(rows):
    print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.0f} | {r[5]} | {r[6]} | {r[7]} | {r[8]:.1f} |")
PY
