#!/bin/bash

set -e

echo "Installing Git, Docker, and Docker Compose on Debian..."

echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y


echo "Installing Docker prerequisites..."
sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release

echo "Adding Docker GPG key..."
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo "Adding Docker repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update

echo "Installing Docker and Docker Compose..."
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "Starting Docker service..."
sudo systemctl start docker
sudo systemctl enable docker

echo "Adding user to docker group..."
sudo usermod -aG docker $USER

echo "Testing installations..."
git --version
docker --version
docker compose version

echo ""
echo "Installation completed successfully!"
echo ""
echo "IMPORTANT: Log out and log back in or restart for Docker permissions to take effect."
echo ""
echo "Quick commands:"
echo "   Git: git --version"
echo "   Docker: docker run hello-world"
echo "   Docker Compose: docker compose --help"
echo ""
echo "Ready to go!"