#!/usr/bin/env bash
set -euo pipefail

# Required env
: "${GCP_PROJECT:?GCP_PROJECT is required}"
: "${REGION:?REGION is required}"                 # e.g. asia-northeast3
: "${SERVICE_NAME:=research-mate-backend}"
: "${INSTANCE_CONNECTION_NAME:?INSTANCE_CONNECTION_NAME is required}"  # project:region:instance
: "${DB_USER:?DB_USER is required}"
: "${DB_NAME:?DB_NAME is required}"
: "${CORS_ALLOW_ORIGINS:?CORS_ALLOW_ORIGINS is required}"             # https://front.example.com
: "${ALLOWED_HOSTS:?ALLOWED_HOSTS is required}"                       # api.example.com
: "${CPU:=2}"
: "${MEMORY:=2Gi}"
: "${MAX_INSTANCES:=10}"

# Secret names in Secret Manager
: "${SECRET_KEY_SECRET:=research-mate-secret-key}"
: "${DB_PASS_SECRET:=research-mate-db-pass}"
: "${OPENAI_API_KEY_SECRET:=research-mate-openai-key}"
: "${GEMINI_API_KEY_SECRET:=research-mate-gemini-key}"

IMAGE="gcr.io/${GCP_PROJECT}/${SERVICE_NAME}:$(date +%Y%m%d-%H%M%S)"

gcloud builds submit --project "$GCP_PROJECT" --tag "$IMAGE" .

gcloud run deploy "$SERVICE_NAME" \
  --project "$GCP_PROJECT" \
  --region "$REGION" \
  --platform managed \
  --image "$IMAGE" \
  --allow-unauthenticated \
  --port 8080 \
  --cpu "$CPU" \
  --memory "$MEMORY" \
  --max-instances "$MAX_INSTANCES" \
  --add-cloudsql-instances "$INSTANCE_CONNECTION_NAME" \
  --set-env-vars "PROJECT_NAME=Research-Mate,API_V1_STR=/api/v1,ENVIRONMENT=production,AUTO_CREATE_TABLES=false,DB_USER=${DB_USER},DB_NAME=${DB_NAME},INSTANCE_CONNECTION_NAME=${INSTANCE_CONNECTION_NAME},CORS_ALLOW_ORIGINS=${CORS_ALLOW_ORIGINS},ALLOWED_HOSTS=${ALLOWED_HOSTS},GOOGLE_CLOUD_PROJECT=${GCP_PROJECT},GOOGLE_CLOUD_LOCATION=us-central1,GEMINI_MODEL=gemini-2.0-flash,USE_LANGGRAPH=true,MAX_REPORT_REVISIONS=2,TEXTBOOK_DATA_DIR=app/data/textbook" \
  --set-secrets "SECRET_KEY=${SECRET_KEY_SECRET}:latest,DB_PASS=${DB_PASS_SECRET}:latest,OPENAI_API_KEY=${OPENAI_API_KEY_SECRET}:latest,GEMINI_API_KEY=${GEMINI_API_KEY_SECRET}:latest"

echo "Deployed: ${SERVICE_NAME}"
