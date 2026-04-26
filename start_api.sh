#!/bin/bash
# Start the Take Time Solver API server

echo "Starting Take Time Solver API..."
uvicorn api:app --reload --host 0.0.0.0 --port 8000