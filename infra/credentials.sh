#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'


if [ ! -f ".env" ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    echo -e "${YELLOW}Run ./setup.sh first to generate credentials${NC}"
    exit 1
fi

source .env

echo -e "${BLUE}Infrastructure Credentials${NC}"
echo -e "${BLUE}=========================${NC}\n"

echo -e "${GREEN}Traefik Dashboard:${NC}"
echo -e "  URL: https://traefik.${DOMAIN}"
echo -e "  Username: ${TRAEFIK_AUTH_USER}"
echo -e "  Password: ${TRAEFIK_AUTH_PASSWORD}"

echo -e "\n${GREEN}PostgreSQL Database:${NC}"
echo -e "  Host: postgres.${DOMAIN}"
echo -e "  Port: 5432"
echo -e "  Database: ${POSTGRES_DB}"
echo -e "  Username: ${POSTGRES_USER}"
echo -e "  Password: ${POSTGRES_PASSWORD}"

echo -e "\n${GREEN}MinIO/S3 Storage:${NC}"
echo -e "  Console: https://minio.${DOMAIN}"
echo -e "  API: https://minio-api.${DOMAIN}"
echo -e "  Access Key: ${AWS_ACCESS_KEY_ID}"
echo -e "  Secret Key: ${AWS_SECRET_ACCESS_KEY}"

echo -e "\n${GREEN}MLflow Tracking:${NC}"
echo -e "  URL: https://mlflow.${DOMAIN}"
echo -e "  Backend: PostgreSQL"
echo -e "  Artifacts: MinIO S3"

echo -e "\n${GREEN}Homepage Dashboard:${NC}"
echo -e "  URL: https://${DOMAIN}"

echo -e "\n${GREEN}RustDesk Remote Access:${NC}"
echo -e "  Server: https://rustdesk.${DOMAIN}"
echo -e "  Key: ${RUSTDESK_KEY}"

echo -e "\n${YELLOW}Service Status:${NC}"
docker compose ps 2>/dev/null || echo -e "${RED}Docker compose not running${NC}"
