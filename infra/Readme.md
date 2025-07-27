# Tech Infrastructure

Self-hosted infrastructure stack with Traefik, MLflow, MinIO, PostgreSQL, and RustDesk.

## Services

- **Homepage Dashboard** - Service overview and system monitoring
- **MLflow Tracking** - ML experiment tracking and model registry
- **MinIO Storage** - S3-compatible object storage
- **PostgreSQL + PostGIS** - Geospatial database with extensions
- **RustDesk Server** - Remote desktop access
- **Traefik Proxy** - Reverse proxy with automatic SSL certificates

## Quick Setup

1. **Clone and configure:**
   ```bash
   git clone <this-repo>
   cd infra
   cp .env.template .env
   cp homepage-config/.env.template homepage-config/.env
   nano .env  # Edit with your domain and credentials
   ```

2. **Run setup:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

## DNS Records Required

Point these A records to your server IP:
- `yourdomain.com`
- `mlflow.yourdomain.com`
- `minio.yourdomain.com`
- `minio-api.yourdomain.com`
- `postgres.yourdomain.com`
- `rustdesk.yourdomain.com`
- `traefik.yourdomain.com`

## Database Connection

Connect to PostgreSQL using DBeaver or any PostgreSQL client:
- **Host:** `postgres.yourdomain.com`
- **Port:** `5432`
- **Database:** `mlflow`
- **SSL Mode:** Required
- **Credentials:** From `.env` file

## Management

```bash
./manage.sh status          # Check service status
./manage.sh logs mlflow # View MLflow logs
./manage.sh restart postgres # Restart database
./manage.sh update          # Pull latest images
./manage.sh backup          # Create backup
```

## System Service

```bash
sudo systemctl start tech-infra    # Start all services
sudo systemctl stop tech-infra     # Stop all services
sudo systemctl status tech-infra   # Check status
```

## Features

- Automatic SSL certificates via Let's Encrypt
- PostgreSQL with PostGIS geospatial extensions
- S3-compatible storage with MinIO
- Homepage dashboard with system monitoring
- Remote desktop server with RustDesk
- Pre-built MLflow image (no local building)
- Secure credential management via .env files
- Systemd integration for auto-start

## Architecture

All services run behind Traefik reverse proxy with automatic SSL termination. Data is persisted in Docker volumes, and the entire stack can be managed via the included management script.