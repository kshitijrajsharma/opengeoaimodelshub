#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${RED}Tech Infrastructure Cleanup${NC}"
echo -e "${RED}===========================${NC}\n"

echo -e "${YELLOW}This script will completely remove all infrastructure components:${NC}"
echo -e "  • Docker containers and images"
echo -e "  • Docker networks and volumes"
echo -e "  • Local data volumes (postgres, minio, traefik, rustdesk)"
echo -e "  • Systemd service"
echo -e "  • Management scripts"
echo -e "  • Log files"
echo -e "\n${GREEN}What will be preserved:${NC}"
echo -e "  • .env file (contains your credentials)"
echo -e "  • .env.template file"
echo -e "  • docker-compose.yml file"
echo -e "  • This cleanup script"

echo -e "\n${RED}WARNING: This action cannot be undone!${NC}"
read -p "Are you sure you want to proceed? (type 'yes' to confirm): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo -e "${YELLOW}Cleanup cancelled.${NC}"
    exit 0
fi

echo -e "\n${YELLOW}Starting cleanup process...${NC}\n"

# Stop and remove all containers
echo -e "${YELLOW}Stopping and removing containers...${NC}"
if [ -f "docker-compose.yml" ]; then
    docker compose down --remove-orphans 2>/dev/null || true
    echo -e "${GREEN}✓ Containers stopped and removed${NC}"
else
    echo -e "${YELLOW}! docker-compose.yml not found, skipping container removal${NC}"
fi

# Remove Docker images related to the project
echo -e "\n${YELLOW}Removing Docker images...${NC}"
IMAGES_TO_REMOVE=(
    "traefik:v3.0"
    "ghcr.io/gethomepage/homepage:latest"
    "ghcr.io/kshitijrajsharma/opengeoaimodelshub/mlflow:latest"
    "minio/minio:latest"
    "postgis/postgis:16-3.4-alpine"
    "rustdesk/rustdesk-server:latest"
    "httpd:2.4-alpine"
)

for image in "${IMAGES_TO_REMOVE[@]}"; do
    if docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "$image"; then
        docker rmi "$image" 2>/dev/null || true
        echo -e "${GREEN}✓ Removed image: $image${NC}"
    else
        echo -e "${YELLOW}! Image not found: $image${NC}"
    fi
done

# Remove Docker networks
echo -e "\n${YELLOW}Removing Docker networks...${NC}"
NETWORKS_TO_REMOVE=(
    "traefik-network"
    "mlflow-network"
)

for network in "${NETWORKS_TO_REMOVE[@]}"; do
    if docker network ls --format "{{.Name}}" | grep -q "$network"; then
        docker network rm "$network" 2>/dev/null || true
        echo -e "${GREEN}✓ Removed network: $network${NC}"
    else
        echo -e "${YELLOW}! Network not found: $network${NC}"
    fi
done

# Remove Docker volumes
echo -e "\n${YELLOW}Removing Docker volumes...${NC}"
docker volume prune -f 2>/dev/null || true
echo -e "${GREEN}✓ Removed unused Docker volumes${NC}"

# Remove local data volumes
echo -e "\n${YELLOW}Removing local data volumes...${NC}"
if [ -d "volumes" ]; then
    sudo rm -rf volumes/
    echo -e "${GREEN}✓ Removed volumes/ directory${NC}"
else
    echo -e "${YELLOW}! volumes/ directory not found${NC}"
fi

# Remove log files
echo -e "\n${YELLOW}Removing log files...${NC}"
if [ -d "logs" ]; then
    rm -rf logs/
    echo -e "${GREEN}✓ Removed logs/ directory${NC}"
else
    echo -e "${YELLOW}! logs/ directory not found${NC}"
fi

# Remove backup files
echo -e "\n${YELLOW}Removing backup files...${NC}"
if [ -d "backups" ]; then
    rm -rf backups/
    echo -e "${GREEN}✓ Removed backups/ directory${NC}"
else
    echo -e "${YELLOW}! backups/ directory not found${NC}"
fi

# Remove systemd service
echo -e "\n${YELLOW}Removing systemd service...${NC}"
if systemctl is-enabled tech-infra.service >/dev/null 2>&1; then
    sudo systemctl stop tech-infra.service 2>/dev/null || true
    sudo systemctl disable tech-infra.service 2>/dev/null || true
    sudo rm -f /etc/systemd/system/tech-infra.service
    sudo systemctl daemon-reload
    echo -e "${GREEN}✓ Removed systemd service${NC}"
else
    echo -e "${YELLOW}! Systemd service not found or not enabled${NC}"
fi

# Remove management scripts
echo -e "\n${YELLOW}Removing management scripts...${NC}"
SCRIPTS_TO_REMOVE=(
    "manage.sh"
    "credentials.sh"
)

for script in "${SCRIPTS_TO_REMOVE[@]}"; do
    if [ -f "$script" ]; then
        rm -f "$script"
        echo -e "${GREEN}✓ Removed $script${NC}"
    else
        echo -e "${YELLOW}! Script not found: $script${NC}"
    fi
done

# Final Docker cleanup
echo -e "\n${YELLOW}Performing final Docker cleanup...${NC}"
docker system prune -a -f --volumes 2>/dev/null || true
echo -e "${GREEN}✓ Docker system cleanup completed${NC}"

# Summary
echo -e "\n${GREEN}=== CLEANUP COMPLETE ===\n${NC}"
echo -e "${GREEN}Successfully removed:${NC}"
echo -e "  ✓ All Docker containers, images, networks, and volumes"
echo -e "  ✓ Local data volumes (postgres, minio, traefik, rustdesk)"
echo -e "  ✓ Systemd service registration"
echo -e "  ✓ Management and credential scripts"
echo -e "  ✓ Log and backup files"

echo -e "\n${GREEN}Preserved files:${NC}"
echo -e "  ✓ .env (your credentials are safe)"
echo -e "  ✓ .env.template"
echo -e "  ✓ docker-compose.yml"
echo -e "  ✓ setup.sh and prune.sh"

echo -e "\n${BLUE}To reinstall the infrastructure:${NC}"
echo -e "  1. Run: ./setup.sh"
echo -e "  2. Your credentials in .env will be reused"

echo -e "\n${YELLOW}If you want to start completely fresh:${NC}"
echo -e "  1. Delete .env file: rm .env"
echo -e "  2. Run: ./setup.sh (will generate new credentials)"

echo -e "\n${GREEN}Thank you for using Tech Infrastructure!${NC}"
