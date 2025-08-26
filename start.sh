

#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
uvicorn api:app --host 127.0.0.1 --port 8001