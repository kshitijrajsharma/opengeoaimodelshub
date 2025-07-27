#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

generate_password() {
    local length=${1:-16}
    openssl rand -base64 $length | tr -d "=+/" | cut -c1-$length
}


generate_key() {
    local length=${1:-32}
    openssl rand -hex $length
}

echo -e "${BLUE}Tech Infrastructure Setup${NC}"
echo -e "${BLUE}========================${NC}\n"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}Docker is not running${NC}"
    exit 1
fi

if ! command -v docker compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed${NC}"
    exit 1
fi

if ! command -v openssl &> /dev/null; then
    echo -e "${RED}OpenSSL is required for generating secure credentials${NC}"
    exit 1
fi

echo -e "${GREEN}Docker and Docker Compose are ready${NC}\n"

if [ ! -f ".env" ]; then
    if [ -f ".env.template" ]; then
        echo -e "${YELLOW}Creating .env from template...${NC}"
        cp .env.template .env
        
        echo -e "${YELLOW}Generating secure credentials...${NC}"
        
        # Generate secure credentials
        POSTGRES_PASSWORD=$(generate_password 20)
        AWS_ACCESS_KEY=$(generate_key 20)
        AWS_SECRET_KEY=$(generate_key 40)
        RUSTDESK_KEY=$(generate_key 16)
        TRAEFIK_PASSWORD=$(generate_password 16)
        
        # Generate Traefik password hash (doubled $ for docker-compose)
        TRAEFIK_HASH=$(docker run --rm httpd:2.4-alpine htpasswd -nbB admin "$TRAEFIK_PASSWORD" 2>/dev/null | cut -d ":" -f 2 | sed 's/\$/\$\$/g')
        
        # Replace unique placeholders in .env file
        sed -i "s|replace-with-postgres-user|postgres|g" .env
        sed -i "s|replace-with-postgres-password|$POSTGRES_PASSWORD|g" .env
        sed -i "s|replace-with-aws-access-key|$AWS_ACCESS_KEY|g" .env
        sed -i "s|replace-with-aws-secret-key|$AWS_SECRET_KEY|g" .env
        sed -i "s|replace-with-traefik-password|$TRAEFIK_PASSWORD|g" .env
        sed -i "s|replace-with-traefik-hash|$TRAEFIK_HASH|g" .env
        sed -i "s|replace-with-rustdesk-key|$RUSTDESK_KEY|g" .env
        
        echo -e "${GREEN}✓ Generated PostgreSQL password${NC}"
        echo -e "${GREEN}✓ Generated MinIO/AWS credentials${NC}"
        echo -e "${GREEN}✓ Generated Traefik authentication${NC}"
        echo -e "${GREEN}✓ Generated RustDesk key${NC}"

        echo -e "\n${BLUE}IMPORTANT: Save these credentials securely!${NC}"
        echo -e "${YELLOW}Traefik Dashboard Credentials:${NC}"
        echo -e "  Username: admin"
        echo -e "  Password: $TRAEFIK_PASSWORD"
        echo -e "\n${YELLOW}PostgreSQL Database Credentials:${NC}"
        echo -e "  Username: postgres"
        echo -e "  Password: $POSTGRES_PASSWORD"
        echo -e "\n${YELLOW}MinIO/S3 Credentials:${NC}"
        echo -e "  Access Key: $AWS_ACCESS_KEY"
        echo -e "  Secret Key: $AWS_SECRET_KEY"
        echo -e "\n${RED}Please update DOMAIN and ACME_EMAIL in .env before continuing${NC}"
        echo -e "${YELLOW}Run: nano .env${NC}"
        exit 1
    else
        echo -e "${RED}.env file not found and no template available${NC}"
        exit 1
    fi
fi

if [ ! -f "homepage-config/.env" ]; then
    if [ -f "homepage-config/.env.template" ]; then
        echo -e "${YELLOW}Creating homepage .env from template...${NC}"
        cp homepage-config/.env.template homepage-config/.env
    fi
fi

source .env

# Validate that domain and email are set
if [ "$DOMAIN" = "example.com" ] || [ "$ACME_EMAIL" = "admin@example.com" ]; then
    echo -e "${RED}Please update DOMAIN and ACME_EMAIL in .env file${NC}"
    echo -e "${YELLOW}Run: nano .env${NC}"
    exit 1
fi

echo -e "${GREEN}Configuration validated${NC}\n"

echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p volumes/{traefik-data,minio,postgres,rustdesk}
mkdir -p logs

echo -e "${YELLOW}Setting permissions...${NC}"
touch volumes/traefik-data/acme.json
chmod 600 volumes/traefik-data/acme.json
chown -R $USER:$USER volumes/

echo -e "${YELLOW}Stopping any existing services...${NC}"
docker compose down --remove-orphans 2>/dev/null || true

echo -e "${YELLOW}Pulling images from registries...${NC}"
echo -e "${BLUE}Using MLflow image: ${MLFLOW_IMAGE}${NC}"
docker compose pull

echo -e "${YELLOW}Starting services...${NC}"
docker compose up -d

echo -e "${YELLOW}Waiting for services to initialize...${NC}"
sleep 30

echo -e "\n${BLUE}Service Status:${NC}"
docker compose ps

echo -e "\n${YELLOW}Setting up systemd service...${NC}"
sudo tee /etc/systemd/system/tech-infra.service > /dev/null << EOF
[Unit]
Description=Tech Infrastructure Services
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
User=$USER
Group=$USER
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable tech-infra.service

cat > manage.sh << 'EOF'
#!/bin/bash

case "$1" in
    start)
        docker compose up -d
        ;;
    stop)
        docker compose down
        ;;
    restart)
        docker compose restart ${2:-}
        ;;
    logs)
        if [ -z "$2" ]; then
            docker compose logs
        else
            docker compose logs -f "$2"
        fi
        ;;
    status)
        docker compose ps
        ;;
    update)
        echo "Pulling latest images..."
        docker compose pull
        echo "Restarting services with latest images..."
        docker compose up -d
        ;;
    backup)
        BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$BACKUP_DIR"
        cp -r volumes/ "$BACKUP_DIR/"
        docker compose exec -T postgres pg_dump -U ${POSTGRES_USER} ${POSTGRES_DB} > "$BACKUP_DIR/postgres_dump.sql"
        echo "Backup created: $BACKUP_DIR"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|status|update|backup}"
        echo "Examples:"
        echo "  $0 logs mlflow  # View MLflow logs"
        echo "  $0 restart postgres # Restart PostgreSQL"
        echo "  $0 update          # Pull latest images and restart"
        ;;
esac
EOF

chmod +x manage.sh

echo -e "\n${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}Services:${NC}"
echo -e "  Homepage Dashboard: https://${DOMAIN}"
echo -e "  MLflow Tracking: https://mlflow.${DOMAIN}"
echo -e "  MinIO Console: https://minio.${DOMAIN}"
echo -e "  MinIO API: https://minio-api.${DOMAIN}"
echo -e "  RustDesk Server: https://rustdesk.${DOMAIN}"
echo -e "  Traefik Dashboard: https://traefik.${DOMAIN}"
echo -e "  PostgreSQL Database: postgres.${DOMAIN}:5432"

echo -e "\n${GREEN}Management Commands:${NC}"
echo -e "  ./manage.sh status         # Check all services"
echo -e "  ./manage.sh logs mlflow    # View MLflow logs"
echo -e "  ./manage.sh restart postgres # Restart database"
echo -e "  ./manage.sh update         # Pull latest images"
echo -e "  ./manage.sh backup         # Create full backup"

echo -e "\n${BLUE}=== CREDENTIALS ===\n${NC}"
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
echo -e "  Uses PostgreSQL backend and MinIO storage"

echo -e "\n${YELLOW}These credentials are also stored in your .env file${NC}"
echo -e "${RED}IMPORTANT: Keep your .env file secure and backed up!${NC}"
