#!/bin/sh
# Container entrypoint.
#   api      -> serve the HTTP API (default)
#   migrate  -> apply database migrations and exit
#   anything else is executed as-is, e.g. `pulse ingest --query ...`
set -eu

case "${1:-api}" in
  api)
    # One worker per container: the model lives in process memory, so scale by replicas.
    exec uvicorn --factory paper_similarity.api.app:create_app \
      --host 0.0.0.0 --port 8000 --proxy-headers --no-access-log
    ;;
  migrate)
    exec alembic upgrade head
    ;;
  *)
    exec "$@"
    ;;
esac
