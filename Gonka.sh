#!/bin/bash

# Docker One-Click Installation Script (Ubuntu)
# Set color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if running as root user and set command prefix
SUDO_CMD=""
check_root() {
    if [ "$EUID" -eq 0 ]; then
        echo -e "${YELLOW}[WARNING] Detected running as root user${NC}"
        echo -e "${YELLOW}[INFO] Recommended to run as regular user, script will use sudo when needed${NC}"
        echo -e "${YELLOW}[INFO] Continuing to run as root user...${NC}"
        SUDO_CMD=""  # root user does not need sudo
        echo ""
    else
        # Check sudo privileges
        if ! sudo -n true 2>/dev/null; then
            echo -e "${YELLOW}[INFO] This script requires sudo privileges, password may be needed${NC}"
            echo ""
        fi
        SUDO_CMD="sudo"  # Regular user needs to use sudo
    fi
}

# Check if system is Ubuntu
check_ubuntu() {
    if [ ! -f /etc/os-release ]; then
        echo -e "${RED}[ERROR] Cannot detect system type${NC}"
        exit 1
    fi
    
    . /etc/os-release
    
    if [ "$ID" != "ubuntu" ]; then
        echo -e "${RED}[ERROR] This script only supports Ubuntu systems${NC}"
        echo -e "${YELLOW}[INFO] Detected system: $ID${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}[INFO] Detected Ubuntu system: $VERSION${NC}"
    echo ""
}

# Check if Docker is installed
check_docker_installed() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Check Docker Installation Status${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    # Check if docker command exists
    if ! command -v docker &> /dev/null; then
        echo -e "${YELLOW}[INFO] Docker not installed, will proceed with installation${NC}"
        echo ""
        return 1
    fi
    
    # Docker is installed, getting version information
    DOCKER_VERSION=$(docker --version 2>/dev/null)
    echo -e "${GREEN}[INFO] Docker is installed: $DOCKER_VERSION${NC}"
    echo ""
    
    # Check Docker Compose
    if docker compose version &> /dev/null; then
        COMPOSE_VERSION=$(docker compose version 2>/dev/null | head -n1)
        echo -e "${GREEN}[INFO] Docker Compose is installed: $COMPOSE_VERSION${NC}"
    else
        echo -e "${YELLOW}[INFO] Docker Compose not detected${NC}"
    fi
    echo ""
    
    # Check Docker service status
    if $SUDO_CMD systemctl is-active --quiet docker 2>/dev/null; then
        echo -e "${GREEN}[INFO] Docker service is running${NC}"
    else
        echo -e "${YELLOW}[INFO] Docker service not running, starting...${NC}"
        if $SUDO_CMD systemctl start docker 2>/dev/null; then
            echo -e "${GREEN}✓ Docker service started successfully${NC}"
        else
            echo -e "${RED}[ERROR] Docker service failed to start${NC}"
        fi
    fi
    
    # Set to start on boot
    if $SUDO_CMD systemctl is-enabled --quiet docker 2>/dev/null; then
        echo -e "${GREEN}[INFO] Docker service is set to start on boot${NC}"
    else
        echo -e "${YELLOW}[INFO] Setting Docker service to start on boot...${NC}"
        $SUDO_CMD systemctl enable docker 2>/dev/null
        echo -e "${GREEN}✓ Docker service has been set to start on boot${NC}"
    fi
    echo ""
    
    # Check if current user is in docker group (skip for root user)
    if [ "$EUID" -eq 0 ]; then
        echo -e "${GREEN}[INFO] root user does not need to be added to docker group${NC}"
    elif groups | grep -q docker; then
        echo -e "${GREEN}[INFO] Current user is already in docker group${NC}"
    else
        echo -e "${YELLOW}[INFO] Current user is not in docker group${NC}"
        read -p "Add current user to docker group? (y/n): " add_user
        if [[ "$add_user" =~ ^[Yy]$ ]]; then
            if $SUDO_CMD usermod -aG docker $USER 2>/dev/null; then
                echo -e "${GREEN}✓ User added to docker group${NC}"
                echo -e "${YELLOW}[INFO] Need to re-login or run 'newgrp docker' for changes to take effect${NC}"
            else
                echo -e "${RED}[ERROR] Failed to add user to docker group${NC}"
            fi
        fi
    fi
    echo ""
    
    # Verify Docker is working properly
    echo -e "${GREEN}[VERIFY] Verifying Docker is working properly...${NC}"
    if $SUDO_CMD docker info &> /dev/null; then
        echo -e "${GREEN}✓ Docker is running normally${NC}"
    else
        echo -e "${YELLOW}[WARNING] Docker may have issues, but installation detected${NC}"
    fi
    echo ""
    
    # Docker is installed and working, continue with verification steps
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}   Docker Already Installed, Skipping Installation Steps${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${CYAN}Docker Status Information:${NC}"
    echo ""
    echo -e "${YELLOW}Docker Version:${NC}${GREEN}$DOCKER_VERSION${NC}"
    if command -v docker &> /dev/null && $SUDO_CMD docker info &> /dev/null; then
        echo -e "${YELLOW}Docker Service:${NC}${GREEN}Running${NC}"
    fi
    echo ""
    
    # Return 0 indicates Docker is installed, continue with verification steps
    return 0
}

# Remove Old Docker Versions
remove_old_docker() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Remove Old Docker Versions${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    echo -e "${GREEN}[STEP] Removing old Docker versions...${NC}"
    $SUDO_CMD apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true
    echo -e "${GREEN}✓ Old version cleanup completed${NC}"
    echo ""
}

# Install Dependencies
install_dependencies() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Install Dependencies${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    echo -e "${GREEN}[STEP] Updating package list...${NC}"
    if $SUDO_CMD apt-get update; then
        echo -e "${GREEN}✓ Package list updated successfully${NC}"
    else
        echo -e "${RED}[ERROR] Package list update failed${NC}"
        exit 1
    fi
    echo ""
    
    echo -e "${GREEN}[STEP] Installing required dependencies...${NC}"
    if $SUDO_CMD apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release; then
        echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
    else
        echo -e "${RED}[ERROR] Dependencies installation failed${NC}"
        exit 1
    fi
    echo ""
}

# Add Docker Official GPG Key
add_docker_gpg_key() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Add Docker Official GPG Key${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    echo -e "${GREEN}[STEP] Creating key directory...${NC}"
    $SUDO_CMD mkdir -p /etc/apt/keyrings
    echo -e "${GREEN}✓ Directory created successfully${NC}"
    echo ""
    
    echo -e "${GREEN}[STEP] Downloading and adding Docker GPG key...${NC}"
    if curl -fsSL https://download.docker.com/linux/ubuntu/gpg | $SUDO_CMD gpg --dearmor -o /etc/apt/keyrings/docker.gpg; then
        echo -e "${GREEN}✓ GPG key added successfully${NC}"
    else
        echo -e "${RED}[ERROR] Failed to add GPG key, please check network connection${NC}"
        exit 1
    fi
    echo ""
    
    # Setting correct permissions
    $SUDO_CMD chmod a+r /etc/apt/keyrings/docker.gpg
    echo -e "${GREEN}✓ Key permissions set successfully${NC}"
    echo ""
}

# Add Docker Repository
add_docker_repository() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Add Docker Official Repository${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    # Get system architecture
    ARCH=$(dpkg --print-architecture)
    CODENAME=$(lsb_release -cs)
    
    echo -e "${GREEN}[INFO] System architecture: $ARCH${NC}"
    echo -e "${GREEN}[INFO] Ubuntu codename: $CODENAME${NC}"
    echo ""
    
    echo -e "${GREEN}[STEP] Adding Docker repository...${NC}"
    if echo \
        "deb [arch=$ARCH signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
        $CODENAME stable" | $SUDO_CMD tee /etc/apt/sources.list.d/docker.list > /dev/null; then
        echo -e "${GREEN}✓ Docker repository added successfully${NC}"
    else
        echo -e "${RED}[ERROR] Failed to add Docker repository${NC}"
        exit 1
    fi
    echo ""
    
    echo -e "${GREEN}[STEP] Updating package list...${NC}"
    if $SUDO_CMD apt-get update; then
        echo -e "${GREEN}✓ Package list updated successfully${NC}"
    else
        echo -e "${RED}[ERROR] Package list update failed${NC}"
        exit 1
    fi
    echo ""
}

# Install Docker
install_docker() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Install Docker${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    echo -e "${GREEN}[STEP] Installing Docker Engine, CLI, Containerd and Docker Compose...${NC}"
    echo -e "${YELLOW}[INFO] This may take a few minutes, please wait...${NC}"
    echo ""
    
    if $SUDO_CMD apt-get install -y \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin; then
        echo ""
        echo -e "${GREEN}✓ Docker installed successfully${NC}"
    else
        echo ""
        echo -e "${RED}[ERROR] Docker installation failed${NC}"
        exit 1
    fi
    echo ""
}

# Configure Docker Service
configure_docker_service() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Configure Docker Service${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    echo -e "${GREEN}[STEP] Starting Docker service...${NC}"
    if $SUDO_CMD systemctl start docker; then
        echo -e "${GREEN}✓ Docker service started successfully${NC}"
    else
        echo -e "${RED}[ERROR] Docker service failed to start${NC}"
        exit 1
    fi
    echo ""
    
    echo -e "${GREEN}[步骤] In progress设置 Docker 服务开机自启...${NC}"
    if $SUDO_CMD systemctl enable docker; then
        echo -e "${GREEN}✓ Docker service has been set to start on boot${NC}"
    else
        echo -e "${YELLOW}[WARNING] Failed to set Docker service to start on boot${NC}"
    fi
    echo ""
    
    # Check service status
    if $SUDO_CMD systemctl is-active --quiet docker; then
        echo -e "${GREEN}✓ Docker service is running normally${NC}"
    else
        echo -e "${RED}[ERROR] Docker service is not running properly${NC}"
        exit 1
    fi
    echo ""
}

# Configure User Permissions
configure_user_permissions() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Configure User Permissions${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    # Check if current user is already in docker group
    if groups | grep -q docker; then
        echo -e "${GREEN}[INFO] Current user is already in docker group${NC}"
    else
        echo -e "${GREEN}[STEP] Adding current user to docker group...${NC}"
        if $SUDO_CMD usermod -aG docker $USER; then
            echo -e "${GREEN}✓ User added to docker group${NC}"
            echo -e "${YELLOW}[INFO] Need to re-login or run 'newgrp docker' to run Docker without sudo${NC}"
        else
            echo -e "${RED}[ERROR] Failed to add user to docker group${NC}"
        fi
    fi
    echo ""
}

# Verify installation
verify_installation() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Verify Docker Installation${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    # Check docker command
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version 2>/dev/null)
        echo -e "${GREEN}✓ Docker command available: $DOCKER_VERSION${NC}"
    else
        echo -e "${RED}[ERROR] Docker command not available${NC}"
        exit 1
    fi
    echo ""
    
    # Check docker compose command
    if docker compose version &> /dev/null; then
        COMPOSE_VERSION=$(docker compose version 2>/dev/null)
        echo -e "${GREEN}✓ Docker Compose available: $COMPOSE_VERSION${NC}"
    else
        echo -e "${YELLOW}[WARNING] Docker Compose not available${NC}"
    fi
    echo ""
    
    # Run test container
    echo -e "${GREEN}[STEP] Running test container to verify installation...${NC}"
    if $SUDO_CMD docker run --rm hello-world &> /dev/null; then
        echo -e "${GREEN}✓ Docker test container ran successfully${NC}"
        echo ""
        $SUDO_CMD docker run --rm hello-world
    else
        echo -e "${YELLOW}[WARNING] Docker test container failed, but Docker may be correctly installed${NC}"
    fi
    echo ""
}

# Verify GPU Support
verify_gpu_support() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Verify GPU Support${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}[INFO] nvidia-smi command not detected${NC}"
        echo -e "${YELLOW}[INFO] If system has NVIDIA GPU, please install NVIDIA drivers first${NC}"
        echo ""
        return 1
    fi
    
    echo -e "${GREEN}[INFO] NVIDIA driver detected${NC}"
    echo ""
    
    # Check if nvidia-container-toolkit is installed
    if ! dpkg -l | grep -q nvidia-container-toolkit; then
        echo -e "${YELLOW}[INFO] nvidia-container-toolkit not detected${NC}"
        echo -e "${YELLOW}[INFO] nvidia-container-toolkit is required for GPU support${NC}"
        read -p "Install nvidia-container-toolkit now? (y/n): " install_nvidia_toolkit
        if [[ "$install_nvidia_toolkit" =~ ^[Yy]$ ]]; then
            install_nvidia_container_toolkit
        else
            echo -e "${YELLOW}[SKIP] Skipping GPU support verification${NC}"
            echo ""
            return 1
        fi
    else
        echo -e "${GREEN}[INFO] nvidia-container-toolkit is installed${NC}"
    fi
    echo ""
    
    # Check if Docker supports --gpus parameter
    echo -e "${GREEN}[VERIFY] Testing Docker GPU support...${NC}"
    
    # Use sudo or run directly, depends on whether user is in docker group or is root
    DOCKER_CMD="docker"
    if [ "$EUID" -ne 0 ] && ! groups | grep -q docker; then
        DOCKER_CMD="sudo docker"
    fi
    
    # Detect Ubuntu version and select appropriate image
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        UBUNTU_VERSION=$(echo "$VERSION_ID" | cut -d. -f1,2)
    else
        UBUNTU_VERSION="20.04"
    fi
    
    # Define multiple available image tags (sorted by priority)
    CUDA_IMAGES=(
        "nvidia/cuda:11.8.0-base-ubuntu${UBUNTU_VERSION}"
        "nvidia/cuda:11.8-base-ubuntu${UBUNTU_VERSION}"
        "nvidia/cuda:11.8.0-base-ubuntu20.04"
        "nvidia/cuda:11.8-base-ubuntu20.04"
        "nvidia/cuda:12.0.0-base-ubuntu${UBUNTU_VERSION}"
        "nvidia/cuda:12.0-base-ubuntu${UBUNTU_VERSION}"
        "nvidia/cuda:latest"
    )
    
    SELECTED_IMAGE=""
    
    # Try to find available image
    echo -e "${GREEN}[STEP] Looking for available CUDA images...${NC}"
    for IMAGE in "${CUDA_IMAGES[@]}"; do
        echo -e "${YELLOW}   Trying image: $IMAGE${NC}"
        # Try to pull image first (if it does not exist)
        if $DOCKER_CMD pull "$IMAGE" &> /dev/null; then
            SELECTED_IMAGE="$IMAGE"
            echo -e "${GREEN}   ✓ Image available: $IMAGE${NC}"
            break
        fi
    done
    
    if [ -z "$SELECTED_IMAGE" ]; then
        echo -e "${YELLOW}[WARNING] Cannot find available CUDA image, trying generic image...${NC}"
        # Try to use latest base image
        if $DOCKER_CMD pull nvidia/cuda:base-ubuntu20.04 &> /dev/null; then
            SELECTED_IMAGE="nvidia/cuda:base-ubuntu20.04"
        elif $DOCKER_CMD pull nvidia/cuda:base &> /dev/null; then
            SELECTED_IMAGE="nvidia/cuda:base"
        else
            echo -e "${RED}[ERROR] Cannot pull any CUDA image${NC}"
            echo -e "${YELLOW}[INFO] Please check network connection or manually pull image${NC}"
            echo ""
            return 1
        fi
    fi
    echo ""
    
    # Test GPU support
    echo -e "${GREEN}[TEST] Testing GPU support (using image: $SELECTED_IMAGE）...${NC}"
    if $DOCKER_CMD run --rm --gpus all "$SELECTED_IMAGE" nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ Docker GPU support verified successfully!${NC}"
        echo ""
        echo -e "${CYAN}GPU Information:${NC}"
        $DOCKER_CMD run --rm --gpus all "$SELECTED_IMAGE" nvidia-smi
        echo ""
        return 0
    else
        echo -e "${YELLOW}[WARNING] Docker GPU support verification failed${NC}"
        echo -e "${YELLOW}[INFO] Possible reasons:${NC}"
        echo -e "${YELLOW}   1. Docker service needs restart${NC}"
        echo -e "${YELLOW}   2. nvidia-container-toolkit not configured correctly${NC}"
        echo -e "${YELLOW}   3. NVIDIA driver not correctly installed${NC}"
        echo -e "${YELLOW}   4. You can try to run manually: $DOCKER_CMD run --rm --gpus all $SELECTED_IMAGE nvidia-smi${NC}"
        echo ""
        return 1
    fi
}

# Install NVIDIA Container Toolkit
install_nvidia_container_toolkit() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Install NVIDIA Container Toolkit${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    # Add GPG key
    echo -e "${GREEN}[STEP 1/4] Adding NVIDIA Container Toolkit GPG key...${NC}"
    if curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | $SUDO_CMD gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null; then
        echo -e "${GREEN}✓ GPG key added successfully${NC}"
    else
        echo -e "${RED}[ERROR] Failed to add GPG key${NC}"
        return 1
    fi
    echo ""
    
    # Add repository
    echo -e "${GREEN}[STEP 2/4] Adding NVIDIA Container Toolkit repository...${NC}"
    if curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       $SUDO_CMD tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null; then
        echo -e "${GREEN}✓ Repository added successfully${NC}"
    else
        echo -e "${RED}[ERROR] Failed to add repository${NC}"
        return 1
    fi
    echo ""
    
    # Update and install
    echo -e "${GREEN}[STEP 3/4] Updating package list and installing nvidia-container-toolkit...${NC}"
    if $SUDO_CMD apt-get update && $SUDO_CMD apt-get install -y nvidia-container-toolkit; then
        echo -e "${GREEN}✓ nvidia-container-toolkit installation completed${NC}"
    else
        echo -e "${RED}[ERROR] nvidia-container-toolkit installation failed${NC}"
        return 1
    fi
    echo ""
    
    # Configure Docker runtime
    echo -e "${GREEN}[STEP 4/4] Configuring Docker GPU runtime...${NC}"
    if $SUDO_CMD nvidia-ctk runtime configure --runtime=docker; then
        echo -e "${GREEN}✓ Docker GPU runtime configured successfully${NC}"
    else
        echo -e "${RED}[ERROR] Failed to configure Docker GPU runtime${NC}"
        return 1
    fi
    echo ""
    
    # Restart Docker
    echo -e "${GREEN}[RESTART] Restarting Docker service...${NC}"
    if $SUDO_CMD systemctl restart docker; then
        echo -e "${GREEN}✓ Docker service restarted${NC}"
        echo ""
        sleep 2
    else
        echo -e "${RED}[ERROR] Failed to restart Docker service${NC}"
        return 1
    fi
}

# Install and Verify HuggingFace CLI
install_and_verify_huggingface() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Install HuggingFace CLI${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    # Check if Python is installed
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        echo -e "${YELLOW}[INFO] Python not detected, installing...${NC}"
        if $SUDO_CMD apt-get update && $SUDO_CMD apt-get install -y python3 python3-pip; then
            echo -e "${GREEN}✓ Python installation completed${NC}"
        else
            echo -e "${RED}[ERROR] Python installation failed${NC}"
            echo -e "${YELLOW}[SKIP] Skipping HuggingFace CLI installation${NC}"
            echo ""
            return 1
        fi
        echo ""
    fi
    
    # Determine pip command
    PIP_CMD="pip3"
    if command -v pip &> /dev/null; then
        PIP_CMD="pip"
    elif command -v pip3 &> /dev/null; then
        PIP_CMD="pip3"
    else
        echo -e "${YELLOW}[INFO] pip not detected, installing...${NC}"
        if $SUDO_CMD apt-get install -y python3-pip; then
            PIP_CMD="pip3"
            echo -e "${GREEN}✓ pip installation completed${NC}"
        else
            echo -e "${RED}[ERROR] pip installation failed${NC}"
            echo -e "${YELLOW}[SKIP] Skipping HuggingFace CLI installation${NC}"
            echo ""
            return 1
        fi
    fi
    echo ""
    
    # Check if huggingface_hub is installed
    if $PIP_CMD show huggingface_hub &> /dev/null; then
        HF_VERSION=$($PIP_CMD show huggingface_hub 2>/dev/null | grep "^Version:" | awk '{print $2}')
        echo -e "${GREEN}[INFO] HuggingFace Hub is installed: version $HF_VERSION${NC}"
        echo ""
    else
        echo -e "${GREEN}[STEP] Installing huggingface_hub...${NC}"
        if $PIP_CMD install huggingface_hub; then
            echo -e "${GREEN}✓ HuggingFace Hub installation completed${NC}"
        else
            echo -e "${RED}[ERROR] HuggingFace Hub installation failed${NC}"
            echo -e "${YELLOW}[INFO] You can manually run later: $PIP_CMD install huggingface_hub${NC}"
            echo ""
            return 1
        fi
        echo ""
    fi
    
    # Verify installation
    echo -e "${GREEN}[VERIFY] Verifying HuggingFace CLI...${NC}"
    if python3 -c "import huggingface_hub; print('HuggingFace Hub version:', huggingface_hub.__version__)" 2>/dev/null; then
        HF_VERSION=$(python3 -c "import huggingface_hub; print(huggingface_hub.__version__)" 2>/dev/null)
        echo -e "${GREEN}✓ HuggingFace Hub verified successfully: version $HF_VERSION${NC}"
        echo ""
        
        # Check huggingface-cli command (multiple methods)
        HF_CLI_AVAILABLE=false
        HF_CLI_CMD=""
        
        # Method 1: Check for huggingface-cli in PATH
        if command -v huggingface-cli &> /dev/null; then
            HF_CLI_CMD="huggingface-cli"
            HF_CLI_AVAILABLE=true
        # Method 2: Try using Python module method
        elif python3 -m huggingface_hub.cli --help &> /dev/null; then
            HF_CLI_CMD="python3 -m huggingface_hub.cli"
            HF_CLI_AVAILABLE=true
        # Method 3: Find command in user local bin directory
        elif [ -f "$HOME/.local/bin/huggingface-cli" ]; then
            HF_CLI_CMD="$HOME/.local/bin/huggingface-cli"
            HF_CLI_AVAILABLE=true
        # Method 4: Find script in Python site-packages
        else
            HF_CLI_PATH=$(python3 -c "import site; import os; print(os.path.join(site.getuserbase(), 'bin', 'huggingface-cli'))" 2>/dev/null)
            if [ -f "$HF_CLI_PATH" ]; then
                HF_CLI_CMD="$HF_CLI_PATH"
                HF_CLI_AVAILABLE=true
            fi
        fi
        
        if [ "$HF_CLI_AVAILABLE" = true ]; then
            echo -e "${GREEN}✓ HuggingFace CLI command available${NC}"
            if [ "$HF_CLI_CMD" != "huggingface-cli" ]; then
                echo -e "${YELLOW}[INFO] Use command: $HF_CLI_CMD${NC}"
                # Try to create symbolic link to /usr/local/bin (if possible)
                if [ "$EUID" -eq 0 ] || sudo -n true 2>/dev/null; then
                    if [ -f "$HF_CLI_CMD" ] && [ ! -f "/usr/local/bin/huggingface-cli" ]; then
                        echo -e "${GREEN}[STEP] Creating system-level symbolic link...${NC}"
                        $SUDO_CMD ln -sf "$HF_CLI_CMD" /usr/local/bin/huggingface-cli 2>/dev/null && \
                            echo -e "${GREEN}✓ Symbolic link created, can now use 'huggingface-cli' command directly${NC}" || \
                            echo -e "${YELLOW}[INFO] Symbolic link creation failed, please use: $HF_CLI_CMD${NC}"
                    fi
                fi
            fi
            echo ""
        else
            echo -e "${YELLOW}[INFO] HuggingFace CLI command not in PATH, but library is installed${NC}"
            echo -e "${YELLOW}[INFO] Can be invoked using the following methods:${NC}"
            echo -e "${YELLOW}   - Python module: python3 -m huggingface_hub.cli${NC}"
            echo -e "${YELLOW}   - Python import: python3 -c 'import huggingface_hub'${NC}"
            echo ""
            # Try to install CLI entry point
            echo -e "${GREEN}[STEP] Trying to install CLI entry point...${NC}"
            if $PIP_CMD install --upgrade --force-reinstall huggingface_hub &> /dev/null; then
                if command -v huggingface-cli &> /dev/null; then
                    echo -e "${GREEN}✓ HuggingFace CLI command now available${NC}"
                else
                    echo -e "${YELLOW}[提示] 请使用: python3 -m huggingface_hub.cli${NC}"
                fi
            fi
            echo ""
        fi
        
        return 0
    else
        echo -e "${YELLOW}[WARNING] HuggingFace Hub verification failed${NC}"
        echo -e "${YELLOW}[INFO] You can try to reinstall: $PIP_CMD install --upgrade huggingface_hub${NC}"
        echo ""
        return 1
    fi
}

# Get huggingface-cli command (helper function)
get_huggingface_cli_cmd() {
    # Method 1: Check for huggingface-cli in PATH
    if command -v huggingface-cli &> /dev/null; then
        echo "huggingface-cli"
        return 0
    fi
    
    # 方式2: 查找用户本地 bin 目录中的命令
    if [ -f "$HOME/.local/bin/huggingface-cli" ]; then
        echo "$HOME/.local/bin/huggingface-cli"
        return 0
    fi
    
    # 方式3: 查找 Python site-packages 中的脚本
    HF_CLI_PATH=$(python3 -c "import site; import os; print(os.path.join(site.getuserbase(), 'bin', 'huggingface-cli'))" 2>/dev/null)
    if [ -n "$HF_CLI_PATH" ] && [ -f "$HF_CLI_PATH" ]; then
        echo "$HF_CLI_PATH"
        return 0
    fi
    
    # If none found, return empty (will use Python API method)
    return 1
}

# Download Deployment Files
download_deployment_files() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Download Deployment Files${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    # Save current working directory
    ORIGINAL_DIR=$(pwd)
    
    # Check if git is installed
    if ! command -v git &> /dev/null; then
        echo -e "${YELLOW}[INFO] git not detected, installing...${NC}"
        if $SUDO_CMD apt-get update && $SUDO_CMD apt-get install -y git; then
            echo -e "${GREEN}✓ git installation completed${NC}"
        else
            echo -e "${RED}[ERROR] git installation failed${NC}"
            echo -e "${YELLOW}[SKIP] Skipping deployment file download${NC}"
            echo ""
            return 1
        fi
        echo ""
    fi
    
    # Check if gonka directory already exists
    if [ -d "gonka" ]; then
        echo -e "${YELLOW}[INFO] gonka directory already exists${NC}"
        read -p "Re-clone repository? (y/n): " reclone
        if [[ "$reclone" =~ ^[Yy]$ ]]; then
            echo -e "${GREEN}[STEP] Removing old directory...${NC}"
            rm -rf gonka
            echo -e "${GREEN}✓ Old directory removed${NC}"
            echo ""
        else
            echo -e "${GREEN}[INFO] Using existing directory${NC}"
            echo ""
        fi
    fi
    
    # Clone repository
    if [ ! -d "gonka" ]; then
        echo -e "${GREEN}[STEP] Cloning gonka repository...${NC}"
        if git clone https://github.com/gonka-ai/gonka.git -b main; then
            echo -e "${GREEN}✓ Repository cloned successfully${NC}"
        else
            echo -e "${RED}[ERROR] Repository clone failed${NC}"
            echo -e "${YELLOW}[INFO] Please check network connection or GitHub access${NC}"
            echo ""
            return 1
        fi
        echo ""
    fi
    
    # Enter deployment directory
    if [ -d "gonka/deploy/join" ]; then
        echo -e "${GREEN}[STEP] Entering deployment directory...${NC}"
        cd gonka/deploy/join || {
            echo -e "${RED}[ERROR] Cannot enter deployment directory${NC}"
            cd "$ORIGINAL_DIR"
            return 1
        }
        echo -e "${GREEN}✓ Entered deployment directory: $(pwd)${NC}"
        echo ""
        
        # Copy configuration file template
        if [ -f "config.env.template" ]; then
            if [ ! -f "config.env" ]; then
                echo -e "${GREEN}[STEP] Copying configuration file template...${NC}"
                if cp config.env.template config.env; then
                    echo -e "${GREEN}✓ Configuration file created: $(pwd)/config.env${NC}"
                    echo -e "${YELLOW}[INFO] Please edit config.env file for configuration${NC}"
                else
                    echo -e "${RED}[ERROR] Configuration file copy failed${NC}"
                    cd "$ORIGINAL_DIR"
                    return 1
                fi
            else
                echo -e "${GREEN}[INFO] Configuration file config.env already exists${NC}"
                echo -e "${YELLOW}[INFO] To reconfigure, delete existing config.env file${NC}"
            fi
        else
            echo -e "${YELLOW}[WARNING] config.env.template file not found${NC}"
        fi
        echo ""
        
        # Return to original directory
        cd "$ORIGINAL_DIR"
        
        return 0
    else
        echo -e "${YELLOW}[WARNING] Deployment directory gonka/deploy/join not found${NC}"
        echo -e "${YELLOW}[INFO] Repository structure may have changed${NC}"
        echo ""
        return 1
    fi
}

# Download and Install inferenced Binary
install_inferenced() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}    Downloading and Installing inferenced${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    # Check if necessary tools are installed
    MISSING_TOOLS=()
    if ! command -v wget &> /dev/null; then MISSING_TOOLS+=("wget"); fi
    if ! command -v unzip &> /dev/null; then MISSING_TOOLS+=("unzip"); fi
    if ! command -v curl &> /dev/null; then MISSING_TOOLS+=("curl"); fi
    
    if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
        echo -e "${YELLOW}[INFO] Following tools not detected: ${MISSING_TOOLS[*]}，installing...${NC}"
        if $SUDO_CMD apt-get update && $SUDO_CMD apt-get install -y "${MISSING_TOOLS[@]}"; then
            echo -e "${GREEN}✓ Tools installation completed${NC}"
        else
            echo -e "${RED}[ERROR] Tools installation failed${NC}"
            echo -e "${YELLOW}[SKIP] Skipping inferenced installation${NC}"
            echo ""
            return 1
        fi
        echo ""
    fi
    
    # Save current working directory
    ORIGINAL_DIR=$(pwd)
    
    # Determine extraction directory (gonka directory in root user home)
    if [ "$EUID" -eq 0 ]; then
        EXTRACT_DIR="/root/gonka"
    else
        EXTRACT_DIR="$HOME/gonka"
    fi
    
    # Determine installation directory (prefer /usr/local/bin, requires root privilege)
    INSTALL_DIR="/usr/local/bin"
    BINARY_NAME="inferenced"
    ZIP_FILE="inferenced-linux-amd64.zip"
    
    # Get latest version download link
    echo -e "${GREEN}[STEP] Getting latest version information...${NC}"
    LATEST_URL=$(curl -s https://api.github.com/repos/gonka-ai/gonka/releases/latest | grep "browser_download_url.*inferenced-linux-amd64.zip" | cut -d '"' -f 4)
    
    if [ -z "$LATEST_URL" ]; then
        # If API fetch fails, use latest tag
        DOWNLOAD_URL="https://github.com/gonka-ai/gonka/releases/latest/download/inferenced-linux-amd64.zip"
        echo -e "${YELLOW}[INFO] Using default download link${NC}"
    else
        DOWNLOAD_URL="$LATEST_URL"
        echo -e "${GREEN}✓ Got latest version download link${NC}"
    fi
    echo ""
    
    # Check if already installed
    if command -v "$BINARY_NAME" &> /dev/null; then
        INSTALLED_PATH=$(which "$BINARY_NAME")
        echo -e "${GREEN}[INFO] inferenced is installed: $INSTALLED_PATH${NC}"
        read -p "Re-download and install? (y/n): " reinstall
        if [[ ! "$reinstall" =~ ^[Yy]$ ]]; then
            echo -e "${GREEN}[INFO] Skipping installation, using existing version${NC}"
            echo ""
            return 0
        fi
        echo ""
    fi
    
    # Create extraction directory
    echo -e "${GREEN}[STEP] Creating extraction directory: $EXTRACT_DIR${NC}"
    if [ ! -d "$EXTRACT_DIR" ]; then
        mkdir -p "$EXTRACT_DIR"
        echo -e "${GREEN}✓ Directory created${NC}"
    else
        echo -e "${GREEN}✓ Directory already exists${NC}"
    fi
    echo ""
    
    # Switch to extraction directory
    cd "$EXTRACT_DIR" || {
        echo -e "${RED}[ERROR] Cannot enter extraction directory: $EXTRACT_DIR${NC}"
        return 1
    }
    echo -e "${GREEN}✓ Switched to extraction directory: $(pwd)${NC}"
    echo ""
    
    # Download zip file
    echo -e "${GREEN}[STEP] Downloading latest version of inferenced zip file...${NC}"
    if wget -q --show-progress "$DOWNLOAD_URL" -O "$ZIP_FILE"; then
        echo -e "${GREEN}✓ Download successful: $EXTRACT_DIR/$ZIP_FILE${NC}"
    else
        echo -e "${RED}[ERROR] Download failed${NC}"
        echo -e "${YELLOW}[INFO] Please check network connection or GitHub access${NC}"
        cd "$ORIGINAL_DIR"
        return 1
    fi
    echo ""
    
    # Extract zip file
    echo -e "${GREEN}[STEP] Extracting zip file to $EXTRACT_DIR...${NC}"
    if unzip -q -o "$ZIP_FILE" -d "$EXTRACT_DIR"; then
        echo -e "${GREEN}✓ Extraction successful${NC}"
    else
        echo -e "${RED}[ERROR] Extraction failed${NC}"
        cd "$ORIGINAL_DIR"
        return 1
    fi
    echo ""
    
    # Find extracted inferenced file
    BINARY_FILE="$EXTRACT_DIR/$BINARY_NAME"
    if [ ! -f "$BINARY_FILE" ]; then
        # May be in subdirectory
        BINARY_FILE=$(find "$EXTRACT_DIR" -name "$BINARY_NAME" -type f | head -n1)
        if [ -z "$BINARY_FILE" ]; then
            echo -e "${RED}[ERROR] Cannot find extracted inferenced file${NC}"
            cd "$ORIGINAL_DIR"
            return 1
        fi
    fi
    
    echo -e "${GREEN}[INFO] Found binary file: $BINARY_FILE${NC}"
    echo ""
    
    # Add execute permission
    echo -e "${GREEN}[STEP] Adding execute permission...${NC}"
    if chmod +x "$BINARY_FILE"; then
        echo -e "${GREEN}✓ Execute permission added${NC}"
    else
        echo -e "${RED}[ERROR] Failed to add execute permission${NC}"
        cd "$ORIGINAL_DIR"
        return 1
    fi
    echo ""
    
    # Test binary file
    echo -e "${GREEN}[TEST] Testing binary file...${NC}"
    if "$BINARY_FILE" --help &> /dev/null; then
        echo -e "${GREEN}✓ Binary file test successful${NC}"
    else
        echo -e "${YELLOW}[WARNING] Binary file test failed, but will continue installation${NC}"
    fi
    echo ""
    
    # Install to system directory
    echo -e "${GREEN}[STEP] Installing to system directory ($INSTALL_DIR)...${NC}"
    if $SUDO_CMD cp "$BINARY_FILE" "$INSTALL_DIR/$BINARY_NAME"; then
        $SUDO_CMD chmod +x "$INSTALL_DIR/$BINARY_NAME"
        echo -e "${GREEN}✓ inferenced installed to $INSTALL_DIR/$BINARY_NAME${NC}"
    else
        echo -e "${RED}[ERROR] Installation failed${NC}"
        cd "$ORIGINAL_DIR"
        return 1
    fi
    echo ""
    
    # Clean up downloaded zip file
    echo -e "${GREEN}[CLEANUP] Cleaning up downloaded zip file...${NC}"
    rm -f "$ZIP_FILE"
    echo -e "${GREEN}✓ Cleanup completed${NC}"
    echo ""
    
    # Verify installation
    echo -e "${GREEN}[VERIFY] Verifying installation...${NC}"
    if command -v "$BINARY_NAME" &> /dev/null; then
        INSTALLED_VERSION=$($BINARY_NAME --version 2>/dev/null || $BINARY_NAME --help 2>/dev/null | head -n1 || echo "已安装")
        echo -e "${GREEN}✓ inferenced installation successful${NC}"
        echo -e "${GREEN}   Location: $(which $BINARY_NAME)${NC}"
        echo ""
        
        # Show help information
        echo -e "${CYAN}inferenced Usage Help:${NC}"
        $BINARY_NAME --help 2>/dev/null | head -n10 || echo -e "${YELLOW}   运行 '$BINARY_NAME --help' to see full help${NC}"
        echo ""
    else
        echo -e "${YELLOW}[WARNING] inferenced command not found in PATH${NC}"
        echo -e "${YELLOW}[INFO] May need to reload shell or check PATH settings${NC}"
        echo ""
    fi
    
    # Return to original directory
    cd "$ORIGINAL_DIR"
    
    return 0
}

# Show usage information
show_usage_info() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}   Docker Installation Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${CYAN}Usage Instructions:${NC}"
    echo ""
    echo -e "${YELLOW}1. If current user has been added to docker group, need to:${NC}"
    echo -e "   ${GREEN}   - Re-login to system, or${NC}"
    echo -e "   ${GREEN}   - Run command: newgrp docker${NC}"
    echo ""
    echo -e "${YELLOW}2. Verify installation (no sudo needed):${NC}"
    echo -e "   ${GREEN}   docker --version${NC}"
    echo -e "   ${GREEN}   docker ps${NC}"
    echo ""
    echo -e "${YELLOW}3. Run test container:${NC}"
    echo -e "   ${GREEN}   docker run hello-world${NC}"
    echo ""
    echo -e "${YELLOW}4. View Docker information:${NC}"
    echo -e "   ${GREEN}   docker info${NC}"
    echo ""
    echo -e "${YELLOW}5. Test GPU support (if configured):${NC}"
    echo -e "   ${GREEN}   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi${NC}"
    echo -e "   ${GREEN}   or use: docker run --rm --gpus all nvidia/cuda:latest nvidia-smi${NC}"
    echo ""
    echo -e "${YELLOW}6. Use HuggingFace CLI:${NC}"
    echo -e "   ${GREEN}   huggingface-cli --help${NC}"
    echo -e "   ${GREEN}   python3 -c 'import huggingface_hub'${NC}"
    echo ""
    echo -e "${YELLOW}7. Deployment file location:${NC}"
    if [ -d "gonka/deploy/join" ]; then
        echo -e "   ${GREEN}   Deployment directory: $(pwd)/gonka/deploy/join${NC}"
        if [ -f "gonka/deploy/join/config.env" ]; then
            echo -e "   ${GREEN}   Configuration file: $(pwd)/gonka/deploy/join/config.env${NC}"
            echo -e "   ${YELLOW}   Please edit configuration file for custom settings${NC}"
        fi
    else
        echo -e "   ${YELLOW}   Deployment files not downloaded or directory does not exist${NC}"
    fi
    echo ""
    echo -e "${YELLOW}8. Use inferenced:${NC}"
    if command -v inferenced &> /dev/null; then
        INFERENCED_PATH=$(which inferenced)
        echo -e "   ${GREEN}   inferenced 已安装: $INFERENCED_PATH${NC}"
        echo -e "   ${GREEN}   运行: inferenced --help${NC}"
    else
        echo -e "   ${YELLOW}   inferenced not installed or not in PATH${NC}"
    fi
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

# Command 1: Deploy Environment
command1_deploy_environment() {
    clear
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Command 1: Deploy Environment${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    # Execute check steps
    check_root
    check_ubuntu
    
    # Check if Docker is installed
    DOCKER_ALREADY_INSTALLED=false
    if check_docker_installed; then
        # Docker 已安装，设置标志
        DOCKER_ALREADY_INSTALLED=true
    fi
    
    # If Docker is not installed, execute installation process
    if [ "$DOCKER_ALREADY_INSTALLED" = false ]; then
        echo -e "${YELLOW}[INFO] Starting Docker installation...${NC}"
        echo ""
        
        # Execute installation steps
        remove_old_docker
        install_dependencies
        add_docker_gpg_key
        add_docker_repository
        install_docker
        configure_docker_service
        configure_user_permissions
        verify_installation
    fi
    
    # Docker installation completed or already exists, execute verification steps
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Execute Additional Verification and Installation${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    # Verify GPU Support
    verify_gpu_support
    
    # Install and Verify HuggingFace CLI
    install_and_verify_huggingface
    
    # Download Deployment Files
    download_deployment_files
    
    # 下载并安装 inferenced
    install_inferenced
    
    # Show usage information
    show_usage_info
    
    echo ""
    read -p "Press Enter to return to main menu..."
}

# Command 2: Create Wallet
command2_create_wallet() {
    clear
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Command 2: Create Wallet${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    # Save current working directory
    ORIGINAL_DIR=$(pwd)
    
    # Determine gonka directory path
    if [ "$EUID" -eq 0 ]; then
        GONKA_DIR="/root/gonka"
    else
        GONKA_DIR="$HOME/gonka"
    fi
    
    # 检查 gonka 目录是否存在
    if [ ! -d "$GONKA_DIR" ]; then
        echo -e "${RED}[ERROR] gonka directory not found: $GONKA_DIR${NC}"
        echo -e "${YELLOW}[INFO] Please execute Command 1: Deploy Environment first${NC}"
        echo ""
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    echo -e "${GREEN}[INFO] Entering directory: $GONKA_DIR${NC}"
    cd "$GONKA_DIR" || {
        echo -e "${RED}[ERROR] Cannot enter directory: $GONKA_DIR${NC}"
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    }
    echo -e "${GREEN}✓ Entered directory${NC}"
    echo ""
    
    # Check if inferenced is available
    if command -v inferenced &> /dev/null; then
        INFERENCED_CMD="inferenced"
    elif [ -f "./inferenced" ]; then
        INFERENCED_CMD="./inferenced"
    else
        echo -e "${RED}[ERROR] inferenced command not found${NC}"
        echo -e "${YELLOW}[INFO] Please execute Command 1: Deploy Environment first${NC}"
        echo ""
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # Show important notice
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}   EXTREMELY IMPORTANT: Security Notice${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo -e "${RED}Write down the mnemonic phrase and store it in a secure offline location.${NC}"
    echo -e "${RED}This is the ONLY way to recover your account key!${NC}"
    echo ""
    echo -e "${YELLOW}Please ensure:${NC}"
    echo -e "${YELLOW}  1. Write the mnemonic phrase on paper or store it in a secure offline device${NC}"
    echo -e "${YELLOW}  2. Do not store the mnemonic phrase on internet-connected devices or in the cloud${NC}"
    echo -e "${YELLOW}  3. Do not share your mnemonic phrase with anyone${NC}"
    echo ""
    echo -e "${RED}========================================${NC}"
    echo ""
    
    read -p "I understand the above security notice, press Enter to continue creating wallet..."
    echo ""
    
    # Execute wallet creation command
    echo -e "${GREEN}[STEP] Creating wallet...${NC}"
    echo -e "${YELLOW}[INFO] Please enter password and confirm password as prompted${NC}"
    echo ""
    
    if $INFERENCED_CMD keys add gonka-account-key --keyring-backend file; then
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}   Wallet Created Successfully!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo -e "${CYAN}Wallet Information:${NC}"
        echo -e "${GREEN}  Key name: gonka-account-key${NC}"
        echo -e "${GREEN}  Key type: file${NC}"
        echo ""
        echo -e "${RED}Please ensure the mnemonic phrase has been safely stored!${NC}"
        echo ""
    else
        echo ""
        echo -e "${RED}[ERROR] Wallet creation failed${NC}"
        echo -e "${YELLOW}[INFO] Please check error messages and retry${NC}"
        echo ""
    fi
    
    # Return to original directory
    cd "$ORIGINAL_DIR"
    
    echo ""
    read -p "Press Enter to return to main menu..."
}

# Command 3: Configure Environment Variables
command3_configure_env() {
    clear
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Command 3: Configure Environment Variables${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    # Save current working directory
    ORIGINAL_DIR=$(pwd)
    
    # Determine configuration file path
    if [ "$EUID" -eq 0 ]; then
        CONFIG_DIR="/root/gonka/deploy/join"
    else
        CONFIG_DIR="$HOME/gonka/deploy/join"
    fi
    
    CONFIG_FILE="$CONFIG_DIR/config.env"
    
    # Check if configuration file directory exists
    if [ ! -d "$CONFIG_DIR" ]; then
        echo -e "${RED}[ERROR] Configuration directory not found: $CONFIG_DIR${NC}"
        echo -e "${YELLOW}[INFO] Please execute Command 1: Deploy Environment first${NC}"
        echo ""
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # Check if configuration file template exists
    TEMPLATE_FILE="$CONFIG_DIR/config.env.template"
    
    # Read existing configuration file or template
    if [ -f "$CONFIG_FILE" ]; then
        echo -e "${GREEN}[INFO] Existing configuration file detected, will update user configuration items${NC}"
        CONFIG_TEMP=$(mktemp)
        cp "$CONFIG_FILE" "$CONFIG_TEMP"
    elif [ -f "$TEMPLATE_FILE" ]; then
        echo -e "${GREEN}[INFO] Using configuration file template${NC}"
        CONFIG_TEMP=$(mktemp)
        cp "$TEMPLATE_FILE" "$CONFIG_TEMP"
    else
        echo -e "${YELLOW}[INFO] Configuration file template not found, will create new configuration file${NC}"
        CONFIG_TEMP=$(mktemp)
        # Create default template content
        cat > "$CONFIG_TEMP" << 'EOF'
export KEY_NAME=<FILLIN>
export KEYRING_PASSWORD=<FILLIN>
export API_PORT=8000
export API_SSL_PORT=8443
export PUBLIC_URL=http://<HOST>:<PORT>
export P2P_EXTERNAL_ADDRESS=tcp://<HOST>:<PORT>
export ACCOUNT_PUBKEY=<ACCOUNT_PUBKEY_FROM_STEP_ABOVE>
export NODE_CONFIG=./node-config.json
export HF_HOME=/mnt/shared
export SEED_API_URL=http://node2.gonka.ai:8000
export SEED_NODE_RPC_URL=http://node2.gonka.ai:26657
export SEED_NODE_P2P_URL=tcp://node2.gonka.ai:5000
export DAPI_API__POC_CALLBACK_URL=http://api:9100
export DAPI_CHAIN_NODE__URL=http://node:26657
export DAPI_CHAIN_NODE__P2P_URL=http://node:26656
export RPC_SERVER_URL_1=http://node1.gonka.ai:26657
export RPC_SERVER_URL_2=http://node2.gonka.ai:26657
export PORT=8080
export INFERENCE_PORT=5050
export KEYRING_BACKEND=file
EOF
    fi
    echo ""
    
    echo -e "${GREEN}[STEP] Starting to configure environment variables...${NC}"
    echo ""
    
    # 1. Node name（可选，default为node）
    echo -e "${CYAN}1. Node name (KEY_NAME)${NC}"
    echo -e "${YELLOW}   Optional, defaults to node${NC}"
    read -p "Please enter node name [default: node]: " KEY_NAME
    if [ -z "$KEY_NAME" ]; then
        KEY_NAME="node"
    fi
    echo -e "${GREEN}   ✓ Node name: $KEY_NAME${NC}"
    echo ""
    
    # 2. Server ML operation password (required)
    echo -e "${CYAN}2. Server ML operation password (KEYRING_PASSWORD)${NC}"
    echo -e "${YELLOW}   Required项${NC}"
    while [ -z "$KEYRING_PASSWORD" ]; do
        read -sp "Please enter server ML operation password: " KEYRING_PASSWORD
        echo ""
        if [ -z "$KEYRING_PASSWORD" ]; then
            echo -e "${RED}   [ERROR] Password cannot be empty, please re-enter${NC}"
        fi
    done
    echo -e "${GREEN}   ✓ Password set${NC}"
    echo ""
    
    # 3. API port（default8000，可更改）
    echo -e "${CYAN}3. API port (API_PORT)${NC}"
    echo -e "${YELLOW}   Default port: 8000${NC}"
    read -p "Please enter API port [default: 8000]: " API_PORT
    if [ -z "$API_PORT" ]; then
        API_PORT="8000"
    fi
    # Verify port is numeric
    if ! [[ "$API_PORT" =~ ^[0-9]+$ ]]; then
        echo -e "${RED}   [ERROR] Port must be numeric, using default value 8000${NC}"
        API_PORT="8000"
    fi
    echo -e "${GREEN}   ✓ API port: $API_PORT${NC}"
    echo ""
    
    # 4. Server public IP（用于PUBLIC_URL和P2P_EXTERNAL_ADDRESS）
    echo -e "${CYAN}4. Server public IP (HOST)${NC}"
    echo -e "${YELLOW}   用于 PUBLIC_URL 和 P2P_EXTERNAL_ADDRESS${NC}"
    while [ -z "$PUBLIC_HOST" ]; do
        read -p "Please enter server public IP: " PUBLIC_HOST
        if [ -z "$PUBLIC_HOST" ]; then
            echo -e "${RED}   [ERROR] IP address cannot be empty, please re-enter${NC}"
        fi
    done
    echo -e "${GREEN}   ✓ Server public IP: $PUBLIC_HOST${NC}"
    echo ""
    
    # 4.1. PUBLIC_URL port
    echo -e "${CYAN}4.1. PUBLIC_URL port${NC}"
    echo -e "${YELLOW}   用于 PUBLIC_URL${NC}"
    echo -e "${RED}   [REQUIRED] Must be filled, cannot use default value${NC}"
    while [ -z "$PUBLIC_URL_PORT" ]; do
        read -p "Please enter PUBLIC_URL port: " PUBLIC_URL_PORT
        if [ -z "$PUBLIC_URL_PORT" ]; then
            echo -e "${RED}   [ERROR] Port cannot be empty, please re-enter${NC}"
        elif ! [[ "$PUBLIC_URL_PORT" =~ ^[0-9]+$ ]]; then
            echo -e "${RED}   [ERROR] Port must be numeric, please re-enter${NC}"
            PUBLIC_URL_PORT=""
        fi
    done
    echo -e "${GREEN}   ✓ PUBLIC_URL port: $PUBLIC_URL_PORT${NC}"
    echo ""
    
    # 4.2. P2P_EXTERNAL_ADDRESS port
    echo -e "${CYAN}4.2. P2P_EXTERNAL_ADDRESS port${NC}"
    echo -e "${YELLOW}   for P2P_EXTERNAL_ADDRESS${NC}"
    echo -e "${RED}   [REQUIRED] Must be filled, cannot use default value${NC}"
    while [ -z "$P2P_PORT" ]; do
        read -p "Please enter P2P_EXTERNAL_ADDRESS port: " P2P_PORT
        if [ -z "$P2P_PORT" ]; then
            echo -e "${RED}   [ERROR] Port cannot be empty, please re-enter${NC}"
        elif ! [[ "$P2P_PORT" =~ ^[0-9]+$ ]]; then
            echo -e "${RED}   [ERROR] Port must be numeric, please re-enter${NC}"
            P2P_PORT=""
        fi
    done
    echo -e "${GREEN}   ✓ P2P_EXTERNAL_ADDRESS port: $P2P_PORT${NC}"
    echo ""
    
    # 5. ACCOUNT_PUBKEY（Required）
    echo -e "${CYAN}5. Account public key (ACCOUNT_PUBKEY)${NC}"
    echo -e "${YELLOW}   Required, obtained from Command 2: Create Wallet step${NC}"
    while [ -z "$ACCOUNT_PUBKEY" ]; do
        read -p "Please enter account public key (ACCOUNT_PUBKEY): " ACCOUNT_PUBKEY
        if [ -z "$ACCOUNT_PUBKEY" ]; then
            echo -e "${RED}   [ERROR] Account public key cannot be empty, please re-enter${NC}"
        fi
    done
    echo -e "${GREEN}   ✓ Account public key set${NC}"
    echo ""
    
    # Update configuration file
    echo -e "${GREEN}[STEP] Updating configuration file...${NC}"
    
    # 使用 awk 安全地替换配置项（避免特殊字符问题）
    awk -v key_name="$KEY_NAME" \
        -v keyring_password="$KEYRING_PASSWORD" \
        -v api_port="$API_PORT" \
        -v public_host="$PUBLIC_HOST" \
        -v public_url_port="$PUBLIC_URL_PORT" \
        -v p2p_port="$P2P_PORT" \
        -v account_pubkey="$ACCOUNT_PUBKEY" \
    'BEGIN {
        updated["KEY_NAME"] = 0
        updated["KEYRING_PASSWORD"] = 0
        updated["API_PORT"] = 0
        updated["PUBLIC_URL"] = 0
        updated["P2P_EXTERNAL_ADDRESS"] = 0
        updated["ACCOUNT_PUBKEY"] = 0
    }
    /^export KEY_NAME=/ {
        print "export KEY_NAME=" key_name
        updated["KEY_NAME"] = 1
        next
    }
    /^export KEYRING_PASSWORD=/ {
        print "export KEYRING_PASSWORD=" keyring_password
        updated["KEYRING_PASSWORD"] = 1
        next
    }
    /^export API_PORT=/ {
        print "export API_PORT=" api_port
        updated["API_PORT"] = 1
        next
    }
    /^export PUBLIC_URL=/ {
        print "export PUBLIC_URL=http://" public_host ":" public_url_port
        updated["PUBLIC_URL"] = 1
        next
    }
    /^export P2P_EXTERNAL_ADDRESS=/ {
        print "export P2P_EXTERNAL_ADDRESS=tcp://" public_host ":" p2p_port
        updated["P2P_EXTERNAL_ADDRESS"] = 1
        next
    }
    /^export ACCOUNT_PUBKEY=/ {
        print "export ACCOUNT_PUBKEY=" account_pubkey
        updated["ACCOUNT_PUBKEY"] = 1
        next
    }
    /^export SEED_API_URL=/ {
        if ($0 ~ /node2\.gonka\.ai:8000/) {
            print "export SEED_API_URL=http://node1.gonka.ai:8000"
        } else {
            print $0
        }
        updated["SEED_API_URL"] = 1
        next
    }
    /^export SEED_NODE_RPC_URL=/ {
        if ($0 ~ /node2\.gonka\.ai:26657/) {
            print "export SEED_NODE_RPC_URL=http://node1.gonka.ai:26657"
        } else {
            print $0
        }
        updated["SEED_NODE_RPC_URL"] = 1
        next
    }
    /^export SEED_NODE_P2P_URL=/ {
        if ($0 ~ /node2\.gonka\.ai:5000/) {
            print "export SEED_NODE_P2P_URL=tcp://node1.gonka.ai:5000"
        } else {
            print $0
        }
        updated["SEED_NODE_P2P_URL"] = 1
        next
    }
    /^export RPC_SERVER_URL_2=/ {
        if ($0 ~ /node2\.gonka\.ai:26657/) {
            print "export RPC_SERVER_URL_2=http://node3.gonka.ai:26657"
        } else {
            print $0
        }
        updated["RPC_SERVER_URL_2"] = 1
        next
    }
    { print }
    END {
        # 如果某些配置项不存在，在file末尾添加
        if (!updated["KEY_NAME"]) {
            print "export KEY_NAME=" key_name
        }
        if (!updated["KEYRING_PASSWORD"]) {
            print "export KEYRING_PASSWORD=" keyring_password
        }
        if (!updated["API_PORT"]) {
            print "export API_PORT=" api_port
        }
        if (!updated["PUBLIC_URL"]) {
            print "export PUBLIC_URL=http://" public_host ":" public_url_port
        }
        if (!updated["P2P_EXTERNAL_ADDRESS"]) {
            print "export P2P_EXTERNAL_ADDRESS=tcp://" public_host ":" p2p_port
        }
        if (!updated["ACCOUNT_PUBKEY"]) {
            print "export ACCOUNT_PUBKEY=" account_pubkey
        }
    }' "$CONFIG_TEMP" > "$CONFIG_TEMP.new" && mv "$CONFIG_TEMP.new" "$CONFIG_TEMP"
    
    # 将更新后的配置写入file
    if cp "$CONFIG_TEMP" "$CONFIG_FILE"; then
        rm -f "$CONFIG_TEMP"
        echo -e "${GREEN}✓ Configuration file updated: $CONFIG_FILE${NC}"
        echo ""
        
        # Show updated configuration content (password hidden)
        echo -e "${CYAN}Updated configuration items:${NC}"
        echo -e "${GREEN}  KEY_NAME=$KEY_NAME${NC}"
        echo -e "${GREEN}  KEYRING_PASSWORD=***(hidden)${NC}"
        echo -e "${GREEN}  API_PORT=$API_PORT${NC}"
        echo -e "${GREEN}  PUBLIC_URL=http://$PUBLIC_HOST:$PUBLIC_URL_PORT${NC}"
        echo -e "${GREEN}  P2P_EXTERNAL_ADDRESS=tcp://$PUBLIC_HOST:$P2P_PORT${NC}"
        echo -e "${GREEN}  ACCOUNT_PUBKEY=$ACCOUNT_PUBKEY${NC}"
        echo ""
        echo -e "${CYAN}Automatically modified configuration items:${NC}"
        echo -e "${GREEN}  SEED_API_URL=http://node1.gonka.ai:8000${NC}"
        echo -e "${GREEN}  SEED_NODE_RPC_URL=http://node1.gonka.ai:26657${NC}"
        echo -e "${GREEN}  SEED_NODE_P2P_URL=tcp://node1.gonka.ai:5000${NC}"
        echo -e "${GREEN}  RPC_SERVER_URL_2=http://node3.gonka.ai:26657${NC}"
        echo ""
        echo -e "${YELLOW}[INFO] Other configuration items retained original values${NC}"
        echo ""
        
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}   Environment Variable Configuration Complete!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
    else
        rm -f "$CONFIG_TEMP"
        echo -e "${RED}[ERROR] Configuration file write failed${NC}"
        echo ""
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # Enter configuration directory
    cd "$CONFIG_DIR" || {
        echo -e "${RED}[ERROR] Cannot enter configuration directory${NC}"
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    }
    
    # Load configuration
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Load configuration${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    echo -e "${GREEN}[STEP] Loading configuration file...${NC}"
    if source config.env; then
        echo -e "${GREEN}✓ Configuration file loaded successfully${NC}"
    else
        echo -e "${RED}[ERROR] Configuration file load failed${NC}"
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    echo ""
    
    # Select Model Configuration
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Select Model Configuration${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    echo -e "${GREEN}Please select model configuration:${NC}"
    echo ""
    echo -e "${YELLOW}  1. Qwen 2.5-7B(default)${NC}"
    echo -e "${YELLOW}  2. QwQ-32 (4 x 3090)${NC}"
    echo ""
    read -p "Please enter option [1-2，default: 1]: " model_choice
    if [ -z "$model_choice" ]; then
        model_choice="1"
    fi
    
    case $model_choice in
        1)
            echo -e "${GREEN}[INFO] Selected Qwen 2.5-7B (default configuration)${NC}"
            MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
            echo ""
            ;;
        2)
            echo -e "${GREEN}[INFO] Selected QwQ-32 (4 x 3090)${NC}"
            MODEL_NAME="Qwen/Qwen3-32B-FP8"
            
            # Check if source file exists
            if [ -f "node-config-qwq-4x3090.json" ]; then
                echo -e "${GREEN}[STEP] Replacing node configuration file...${NC}"
                if cp node-config-qwq-4x3090.json node-config.json; then
                    echo -e "${GREEN}✓ Node configuration file replaced${NC}"
                else
                    echo -e "${RED}[ERROR] Node configuration file replacement failed${NC}"
                    cd "$ORIGINAL_DIR"
                    read -p "Press Enter to return to main menu..."
                    return 1
                fi
            else
                echo -e "${YELLOW}[WARNING] Not found node-config-qwq-4x3090.json file${NC}"
                echo -e "${YELLOW}[INFO] Will use default configuration${NC}"
            fi
            echo ""
            ;;
        *)
            echo -e "${YELLOW}[INFO] Invalid option, using default configuration Qwen 2.5-7B${NC}"
            MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
            echo ""
            ;;
    esac
    
    # Pre-download Model Weights
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Pre-download Model Weights${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    # Ensure in configuration directory
    if [ "$(pwd)" != "$CONFIG_DIR" ]; then
        cd "$CONFIG_DIR" || {
            echo -e "${RED}[ERROR] Cannot enter configuration directory${NC}"
            cd "$ORIGINAL_DIR"
            read -p "Press Enter to return to main menu..."
            return 1
        }
    fi
    
    # Load configuration (reference step: source config.env)
    echo -e "${GREEN}[STEP] Loading configuration file...${NC}"
    if source config.env; then
        echo -e "${GREEN}✓ Configuration file loaded successfully${NC}"
    else
        echo -e "${RED}[ERROR] Configuration file load failed${NC}"
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    echo ""
    
    # Check if HF_HOME is set
    if [ -z "$HF_HOME" ]; then
        echo -e "${YELLOW}[INFO] HF_HOME not set, using default value /mnt/shared${NC}"
        HF_HOME="/mnt/shared"
    fi
    
    # 创建Cache directory
    echo -e "${GREEN}[STEP] Creating cache directory...${NC}"
    if $SUDO_CMD mkdir -p "$HF_HOME"; then
        echo -e "${GREEN}✓ Cache directory created: $HF_HOME${NC}"
    else
        echo -e "${RED}[ERROR] Cache directory creation failed${NC}"
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # 设置目录权限
    echo -e "${GREEN}[STEP] Setting directory permissions...${NC}"
    if $SUDO_CMD chmod 777 "$HF_HOME"; then
        echo -e "${GREEN}✓ Directory permissions set${NC}"
    else
        echo -e "${YELLOW}[WARNING] Directory permissions setting failed, continuing...${NC}"
    fi
    echo ""
    
    # Check if huggingface-cli is available, or use Python API
    HF_CLI_CMD=$(get_huggingface_cli_cmd)
    USE_PYTHON_API=false
    
    if [ -z "$HF_CLI_CMD" ]; then
        echo -e "${YELLOW}[WARNING] huggingface-cli command not available${NC}"
        echo -e "${YELLOW}[STEP] Checking Python API method...${NC}"
        
        # 检查 Python 和 huggingface_hub 是否可用
        if command -v python3 &> /dev/null && python3 -c "import huggingface_hub" &> /dev/null 2>&1; then
            USE_PYTHON_API=true
            echo -e "${GREEN}✓ Will use Python API method to download model${NC}"
        else
            echo -e "${RED}[ERROR] Cannot find HuggingFace CLI or Python module${NC}"
            echo -e "${YELLOW}[INFO] Please execute Command 1: Deploy Environment first${NC}"
            echo -e "${YELLOW}[Tip] Or manually install: pip3 install huggingface_hub${NC}"
            echo ""
            cd "$ORIGINAL_DIR"
            read -p "Press Enter to return to main menu..."
            return 1
        fi
        echo ""
    else
        echo -e "${GREEN}[INFO] Using HuggingFace CLI command: $HF_CLI_CMD${NC}"
        echo ""
    fi
    
    # 下载模型
    echo -e "${GREEN}[步骤] In progress下载模型: $MODEL_NAME${NC}"
    echo -e "${YELLOW}[INFO] This may take a long time, please wait...${NC}"
    echo ""
    
    if [ "$USE_PYTHON_API" = true ]; then
        # Use Python API to download model
        if python3 << EOF
import sys
from huggingface_hub import snapshot_download

try:
    print(f"In progress下载模型: $MODEL_NAME")
    print(f"Cache directory: $HF_HOME")
    snapshot_download(
        repo_id="$MODEL_NAME",
        cache_dir="$HF_HOME",
        local_files_only=False
    )
    print("Model Download Complete!")
    sys.exit(0)
except Exception as e:
    print(f"下载失败: {e}", file=sys.stderr)
    sys.exit(1)
EOF
        then
            echo ""
            echo -e "${GREEN}========================================${NC}"
            echo -e "${GREEN}   Model Download Complete!${NC}"
            echo -e "${GREEN}========================================${NC}"
            echo ""
            echo -e "${CYAN}Model Information:${NC}"
            echo -e "${GREEN}  Model name: $MODEL_NAME${NC}"
            echo -e "${GREEN}  Cache directory: $HF_HOME${NC}"
            echo ""
        else
            echo ""
            echo -e "${RED}[ERROR] Model download failed${NC}"
            echo -e "${YELLOW}[INFO] Please check network connection or retry later${NC}"
            echo ""
        fi
    else
        # Use CLI command to download model
        if $HF_CLI_CMD download "$MODEL_NAME" --cache-dir "$HF_HOME"; then
            echo ""
            echo -e "${GREEN}========================================${NC}"
            echo -e "${GREEN}   Model Download Complete!${NC}"
            echo -e "${GREEN}========================================${NC}"
            echo ""
            echo -e "${CYAN}Model Information:${NC}"
            echo -e "${GREEN}  Model name: $MODEL_NAME${NC}"
            echo -e "${GREEN}  Cache directory: $HF_HOME${NC}"
            echo ""
        else
            echo ""
            echo -e "${RED}[ERROR] Model download failed${NC}"
            echo -e "${YELLOW}[INFO] Please check network connection or retry later${NC}"
            echo ""
        fi
    fi
    
    # Pull Docker Images
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Pull Docker Images${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    # 确保在正确的目录中
    if [ "$(pwd)" != "$CONFIG_DIR" ]; then
        cd "$CONFIG_DIR" || {
            echo -e "${RED}[ERROR] Cannot enter configuration directory${NC}"
            cd "$ORIGINAL_DIR"
            read -p "Press Enter to return to main menu..."
            return 1
        }
    fi
    
    # 检查 docker 是否可用
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}[ERROR] Docker not installed${NC}"
        echo -e "${YELLOW}[INFO] Please execute Command 1: Deploy Environment first${NC}"
        echo ""
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # 检查 docker-compose.yml file是否存在
    if [ ! -f "docker-compose.yml" ]; then
        echo -e "${RED}[ERROR] docker-compose.yml file not found${NC}"
        echo -e "${YELLOW}[INFO] Please ensure in correct directory: $CONFIG_DIR${NC}"
        echo ""
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # 检查 docker-compose.mlnode.yml file是否存在
    if [ ! -f "docker-compose.mlnode.yml" ]; then
        echo -e "${YELLOW}[WARNING] docker-compose.mlnode.yml file not found${NC}"
        echo -e "${YELLOW}[INFO] Will only use docker-compose.yml${NC}"
        COMPOSE_FILES="-f docker-compose.yml"
    else
        COMPOSE_FILES="-f docker-compose.yml -f docker-compose.mlnode.yml"
    fi
    
    echo -e "${GREEN}[STEP] Pulling Docker images...${NC}"
    echo -e "${YELLOW}[INFO] This may take a long time, please wait...${NC}"
    echo ""
    
    # Use sudo or run directly, depends on whether user is in docker group or is root
    DOCKER_COMPOSE_CMD="docker compose"
    if [ "$EUID" -ne 0 ] && ! groups | grep -q docker; then
        DOCKER_COMPOSE_CMD="sudo docker compose"
    fi
    
    if $DOCKER_COMPOSE_CMD $COMPOSE_FILES pull; then
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}   Docker Images Pull Complete!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
    else
        echo ""
        echo -e "${RED}[ERROR] Docker image pull failed${NC}"
        echo -e "${YELLOW}[INFO] Please check network connection or Docker configuration${NC}"
        echo ""
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # Start Initial Services
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Start Initial Services${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    echo -e "${GREEN}[步骤] In progressStart Initial Services (tmkms, node)...${NC}"
    if source config.env && $DOCKER_COMPOSE_CMD $COMPOSE_FILES up tmkms node -d --no-deps; then
        echo ""
        echo -e "${GREEN}✓ Initial services started successfully${NC}"
        echo ""
    else
        echo ""
        echo -e "${RED}[ERROR] Initial services start failed${NC}"
        echo -e "${YELLOW}[INFO] Please check configuration and Docker status${NC}"
        echo ""
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # 验证服务是否启动
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Verify Service Status${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    echo -e "${GREEN}[STEP] Viewing service logs (observing for 3 seconds)...${NC}"
    echo ""
    
    # 使用 timeout 命令观察日志3秒
    if command -v timeout &> /dev/null; then
        timeout 3 $DOCKER_COMPOSE_CMD $COMPOSE_FILES logs tmkms node -f 2>/dev/null || true
    else
        # 如果没有 timeout 命令，使用 sleep 和后台任务
        $DOCKER_COMPOSE_CMD $COMPOSE_FILES logs tmkms node -f &
        LOG_PID=$!
        sleep 3
        kill $LOG_PID 2>/dev/null || true
    fi
    
    echo ""
    echo -e "${GREEN}✓ Service log viewing completed${NC}"
    echo ""
    
    # Return to original directory
    cd "$ORIGINAL_DIR"
    
    echo ""
    read -p "Press Enter to return to main menu..."
}

# Command 4: Check Sync Status
command4_check_sync_status() {
    clear
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Command 4: Check Sync Status${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    # Save current working directory
    ORIGINAL_DIR=$(pwd)
    
    # 确定配置目录路径
    if [ "$EUID" -eq 0 ]; then
        CONFIG_DIR="/root/gonka/deploy/join"
    else
        CONFIG_DIR="$HOME/gonka/deploy/join"
    fi
    
    # 检查配置目录是否存在
    if [ ! -d "$CONFIG_DIR" ]; then
        echo -e "${RED}[ERROR] Configuration directory not found: $CONFIG_DIR${NC}"
        echo -e "${YELLOW}[INFO] Please execute Command 1: Deploy Environment first${NC}"
        echo ""
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # Enter configuration directory
    cd "$CONFIG_DIR" || {
        echo -e "${RED}[ERROR] Cannot enter configuration directory${NC}"
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    }
    
    # 检查 docker 是否可用
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}[ERROR] Docker not installed${NC}"
        echo -e "${YELLOW}[INFO] Please execute Command 1: Deploy Environment first${NC}"
        echo ""
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # Use sudo or run directly, depends on whether user is in docker group or is root
    DOCKER_COMPOSE_CMD="docker compose"
    if [ "$EUID" -ne 0 ] && ! groups | grep -q docker; then
        DOCKER_COMPOSE_CMD="sudo docker compose"
    fi
    
     echo -e "${GREEN}[Step] Viewing sync status logs...${NC}"
    echo -e "${YELLOW}[Tip] Will display tmkms and node service logs, after 10 seconds you can press any key to return${NC}"
    echo ""
    
    # 显示日志（后台运行）
    $DOCKER_COMPOSE_CMD logs tmkms node -f &
    LOG_PID=$!
    
    # 等待10秒
    sleep 10
    
    # 停止日志显示
    kill $LOG_PID 2>/dev/null || true
    
    echo ""
    echo -e "${GREEN}✓ 日志查看完成${NC}"
    echo ""
    
    # Return to original directory
    cd "$ORIGINAL_DIR"
    
    echo ""
    read -p "按任意键返回主菜单..."
}

# Command 5: Create ML Operation Key
command5_create_ml_key() {
    clear
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Command 5: Create ML Operation Key${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    # Save current working directory
    ORIGINAL_DIR=$(pwd)
    
    # 确定配置目录路径
    if [ "$EUID" -eq 0 ]; then
        CONFIG_DIR="/root/gonka/deploy/join"
    else
        CONFIG_DIR="$HOME/gonka/deploy/join"
    fi
    
    # 检查配置目录是否存在
    if [ ! -d "$CONFIG_DIR" ]; then
        echo -e "${RED}[ERROR] Configuration directory not found: $CONFIG_DIR${NC}"
        echo -e "${YELLOW}[INFO] Please execute Command 1: Deploy Environment first${NC}"
        echo ""
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # 检查Configuration file是否存在
    CONFIG_FILE="$CONFIG_DIR/config.env"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}[ERROR] Configuration file not found: $CONFIG_FILE${NC}"
        echo -e "${YELLOW}[INFO] Please execute Command 3: Configure Environment Variables first${NC}"
        echo ""
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # Enter configuration directory
    cd "$CONFIG_DIR" || {
        echo -e "${RED}[ERROR] Cannot enter configuration directory${NC}"
        read -p "Press Enter to return to main menu..."
        return 1
    }
    
    # Load configuration
    echo -e "${GREEN}[STEP] Loading configuration file...${NC}"
    if source config.env; then
        echo -e "${GREEN}✓ Configuration file loaded successfully${NC}"
    else
        echo -e "${RED}[ERROR] Configuration file load failed${NC}"
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    echo ""
    
    # 检查必要的环境变量
    if [ -z "$KEY_NAME" ] || [ -z "$KEYRING_PASSWORD" ] || [ -z "$ACCOUNT_PUBKEY" ]; then
        echo -e "${RED}[ERROR] Configuration file missing necessary environment variables${NC}"
        echo -e "${YELLOW}[INFO] Please execute Command 3: Configure Environment Variables first${NC}"
        echo ""
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # 检查 DAPI 相关环境变量
    if [ -z "$DAPI_API__PUBLIC_URL" ] || [ -z "$DAPI_CHAIN_NODE__SEED_API_URL" ]; then
        echo -e "${YELLOW}[Warning] Configuration file missing DAPI related environment variables${NC}"
        echo -e "${YELLOW}[INFO] Will use default values or read from configuration file${NC}"
        echo ""
    fi
    
    # 检查 docker 是否可用
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}[ERROR] Docker not installed${NC}"
        echo -e "${YELLOW}[INFO] Please execute Command 1: Deploy Environment first${NC}"
        echo ""
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # 检查 docker-compose.yml file是否存在
    if [ ! -f "docker-compose.yml" ]; then
        echo -e "${RED}[ERROR] docker-compose.yml file not found${NC}"
        echo ""
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # Show important notice
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}   重要提示${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""
    echo -e "${YELLOW}Do not execute this command repeatedly, generate once per server, must be maintained after restart${NC}"
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}   Record ML Operation Key Address${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    
    read -p "I understand the above notice, press Enter to continue creating ML operation key..."
    echo ""
    
    # Use sudo or run directly, depends on whether user is in docker group or is root
    DOCKER_COMPOSE_CMD="docker compose"
    if [ "$EUID" -ne 0 ] && ! groups | grep -q docker; then
        DOCKER_COMPOSE_CMD="sudo docker compose"
    fi
    
    # Extract all environment variables from config.env file and build -e parameters
    ENV_ARGS=""
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip empty lines and comments
        if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
            continue
        fi
        # Process export VAR=value format
        if [[ "$line" =~ ^[[:space:]]*export[[:space:]]+([^=]+)=(.*)$ ]]; then
            VAR_NAME="${BASH_REMATCH[1]// /}"
            VAR_VALUE="${BASH_REMATCH[2]}"
            # Remove possible quotes
            VAR_VALUE="${VAR_VALUE#\"}"
            VAR_VALUE="${VAR_VALUE%\"}"
            VAR_VALUE="${VAR_VALUE#\'}"
            VAR_VALUE="${VAR_VALUE%\'}"
            # Build -e parameter
            ENV_ARGS="$ENV_ARGS -e $VAR_NAME=$VAR_VALUE"
        # Process VAR=value format (without export)
        elif [[ "$line" =~ ^[[:space:]]*([^=]+)=(.*)$ ]]; then
            VAR_NAME="${BASH_REMATCH[1]// /}"
            VAR_VALUE="${BASH_REMATCH[2]}"
            # Remove possible quotes
            VAR_VALUE="${VAR_VALUE#\"}"
            VAR_VALUE="${VAR_VALUE%\"}"
            VAR_VALUE="${VAR_VALUE#\'}"
            VAR_VALUE="${VAR_VALUE%\'}"
            # Build -e parameter
            ENV_ARGS="$ENV_ARGS -e $VAR_NAME=$VAR_VALUE"
        fi
    done < config.env
    
    # Determine seed node address (for host registration)
    if [ -z "$SEED_API_URL" ]; then
        SEED_NODE_ADDRESS="http://node2.gonka.ai:8000"
    else
        SEED_NODE_ADDRESS="$SEED_API_URL"
    fi
    
    # Execute command: Create key
    echo -e "${GREEN}[Step] Creating ML operation key...${NC}"
    echo -e "${YELLOW}[INFO] Entering API container...${NC}"
    echo ""
    
    CREATE_KEY_CMD="printf '%s\n%s\n' \"\$KEYRING_PASSWORD\" \"\$KEYRING_PASSWORD\" | inferenced keys add \"\$KEY_NAME\" --keyring-backend file"
    
    # 先执行创建密钥命令
    if $DOCKER_COMPOSE_CMD run --rm --no-deps -it $ENV_ARGS api /bin/sh -c "$CREATE_KEY_CMD"; then
        echo ""
        echo -e "${GREEN}✓ ML 操作密钥创建成功${NC}"
        echo ""
        
        # Get newly created key address (extract from output)
        echo -e "${GREEN}[STEP] Registering host...${NC}"
        echo ""
        
        # Determine node URL (use PUBLIC_URL, if not available use SEED_API_URL)
        if [ -z "$PUBLIC_URL" ]; then
            if [ -z "$SEED_API_URL" ]; then
                NODE_URL="http://node2.gonka.ai:8000"
            else
                NODE_URL="$SEED_API_URL"
            fi
        else
            NODE_URL="$PUBLIC_URL"
        fi
        
        # 确定种子节点地址
        if [ -z "$SEED_API_URL" ]; then
            SEED_NODE_ADDRESS="http://node2.gonka.ai:8000"
        else
            SEED_NODE_ADDRESS="$SEED_API_URL"
        fi
        
        echo -e "${CYAN}[INFO] Node URL: $NODE_URL${NC}"
        echo -e "${CYAN}[INFO] Seed node address: $SEED_NODE_ADDRESS${NC}"
        echo ""
        
        # Execute host registration command
        # 根据错误信息，命令格式应该是：register-new-participant <node-url> <account-public-key> --node-address <seed-node>
        # <node-url> 应该是公共节点URL，<account-public-key> 是Account public key，--node-address 是种子节点地址
        # Use variable values directly in command, not environment variable references
        REGISTER_CMD="inferenced register-new-participant \"$NODE_URL\" \"$ACCOUNT_PUBKEY\" --node-address \"$SEED_NODE_ADDRESS\""
        
        if $DOCKER_COMPOSE_CMD run --rm --no-deps -it $ENV_ARGS api /bin/sh -c "$REGISTER_CMD"; then
            echo ""
            echo -e "${GREEN}========================================${NC}"
            echo -e "${GREEN}   ML Operation Key Creation and Host Registration Successful!${NC}"
            echo -e "${GREEN}========================================${NC}"
            echo ""
            echo -e "${RED}========================================${NC}"
            echo -e "${RED}   Please Record ML Operation Key Address${NC}"
            echo -e "${RED}========================================${NC}"
            echo ""
            echo -e "${CYAN}Key Information:${NC}"
            echo -e "${GREEN}  Key name: $KEY_NAME${NC}"
            echo -e "${GREEN}  Key type: file${NC}"
            echo -e "${GREEN}  Account public key: $ACCOUNT_PUBKEY${NC}"
            echo ""
            echo -e "${YELLOW}[INFO] Please ensure key address has been recorded, this is important security information${NC}"
            echo ""
        else
            echo ""
            echo -e "${YELLOW}[WARNING] Host registration failed, but key has been created${NC}"
            echo -e "${YELLOW}[INFO] You can manually execute registration command later${NC}"
            echo -e "${YELLOW}[INFO] Registration command: inferenced register-new-participant <node-url> <account-public-key> --node-address <seed-node>${NC}"
            echo ""
            echo -e "${GREEN}✓ ML operation key has been created, please record key address${NC}"
            echo ""
        fi
    else
        echo ""
        echo -e "${RED}[ERROR] ML operation key creation failed${NC}"
        echo -e "${YELLOW}[INFO] Please check error messages and retry${NC}"
        echo ""
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
     # Grant permissions
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Granting Permissions${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    echo -e "${YELLOW}▲ IMPORTANT: Execute this step on the secure local machine where you created the account key${NC}"
    echo ""
    
    # 获取用户输入的热钱包地址
    echo -e "${CYAN}Please enter创建 ML 操作密钥时的热钱包地址：${NC}"
    while [ -z "$ML_OPS_WALLET_ADDRESS" ]; do
        read -p "Hot wallet address: " ML_OPS_WALLET_ADDRESS
        if [ -z "$ML_OPS_WALLET_ADDRESS" ]; then
            echo -e "${RED}   [ERROR] Hot wallet address cannot be empty, please re-enter${NC}"
        fi
    done
    echo -e "${GREEN}   ✓ Hot wallet address: $ML_OPS_WALLET_ADDRESS${NC}"
    echo ""
    
    # 确定 gonka 目录
    if [ "$EUID" -eq 0 ]; then
        GONKA_DIR="/root/gonka"
    else
        GONKA_DIR="$HOME/gonka"
    fi
    
    # 进入 gonka 目录执行命令
    if [ ! -d "$GONKA_DIR" ]; then
        echo -e "${RED}[ERROR] gonka directory not found: $GONKA_DIR${NC}"
        echo -e "${YELLOW}[INFO] Please execute Command 1: Deploy Environment first${NC}"
        echo ""
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    cd "$GONKA_DIR" || {
        echo -e "${RED}[Error] Cannot enter gonka directory${NC}"
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    }
    
    # Check if inferenced is available
    if command -v inferenced &> /dev/null; then
        INFERENCED_CMD="inferenced"
    elif [ -f "./inferenced" ]; then
        INFERENCED_CMD="./inferenced"
    else
        echo -e "${RED}[ERROR] inferenced command not found${NC}"
        echo -e "${YELLOW}[INFO] Please execute Command 1: Deploy Environment first${NC}"
        echo ""
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # Use fixed account key name (created in Command 2)
    ACCOUNT_KEY_NAME="gonka-account-key"
    
    # Prompt user to confirm key exists (avoid command hanging)
    echo -e "${YELLOW}[Tip] Please ensure you have executed Command 2 to create account key '$ACCOUNT_KEY_NAME'${NC}"
    echo ""
    
    # Load configuration (set SEED_API_URL environment variable)
    echo -e "${GREEN}[STEP] Loading configuration (setting environment variables)...${NC}"
    if [ -f "$CONFIG_DIR/config.env" ]; then
        source "$CONFIG_DIR/config.env"
    fi
    
    # Check and set SEED_API_URL
    if [ -z "$SEED_API_URL" ]; then
        # 如果未设置，使用default值
        SEED_API_URL="http://node2.gonka.ai:8000"
        echo -e "${YELLOW}[信息] SEED_API_URL 未设置，使用default值: $SEED_API_URL${NC}"
    fi
    
    export SEED_API_URL
    echo -e "${GREEN}✓ SEED_API_URL: $SEED_API_URL${NC}"
    echo ""
    
    # 执行授权权限命令
    echo -e "${GREEN}[STEP] Executing grant permissions command...${NC}"
    echo ""
    
    # Build node URL (remove trailing slash if present)
    NODE_URL="${SEED_API_URL%/}/chain-rpc/"
    
    echo -e "${CYAN}[信息] 执行命令：${NC}"
    echo -e "${CYAN}  $INFERENCED_CMD tx inference grant-ml-ops-permissions \\${NC}"
    echo -e "${CYAN}    $ACCOUNT_KEY_NAME \\${NC}"
    echo -e "${CYAN}    $ML_OPS_WALLET_ADDRESS \\${NC}"
    echo -e "${CYAN}    --from $ACCOUNT_KEY_NAME \\${NC}"
    echo -e "${CYAN}    --keyring-backend file \\${NC}"
    echo -e "${CYAN}    --gas 2000000 \\${NC}"
    echo -e "${CYAN}    --node $NODE_URL${NC}"
    echo ""
    
     if $INFERENCED_CMD tx inference grant-ml-ops-permissions \
        "$ACCOUNT_KEY_NAME" \
        "$ML_OPS_WALLET_ADDRESS" \
        --from "$ACCOUNT_KEY_NAME" \
        --keyring-backend file \
        --gas 2000000 \
        --node "$NODE_URL"; then
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}   Permission Grant Successful!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo -e "${CYAN}Grant Information:${NC}"
        echo -e "${GREEN}  Account key: $ACCOUNT_KEY_NAME${NC}"
        echo -e "${GREEN}  Hot wallet address: $ML_OPS_WALLET_ADDRESS${NC}"
        echo -e "${GREEN}  Node address: $NODE_URL${NC}"
        echo ""
    else
        echo ""
        echo -e "${RED}[Error] Permission grant failed${NC}"
        echo -e "${YELLOW}[Tip] Please check:${NC}"
        echo -e "${YELLOW}  1. Account key '$ACCOUNT_KEY_NAME' exists (execute Command 2)${NC}"
        echo -e "${YELLOW}  2. Hot wallet address is correct${NC}"
        echo -e "${YELLOW}  3. Network connection is normal${NC}"
        echo -e "${YELLOW}  4. SEED_API_URL is correct${NC}"
        echo ""
    fi
    
    # Return to original directory
    cd "$ORIGINAL_DIR"
    
    echo ""
    read -p "Press Enter to return to main menu..."
}

# 命令6：启动全节点
command6_start_full_node() {
    clear
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   命令5：启动全节点${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    # Save current working directory
    ORIGINAL_DIR=$(pwd)
    
    # 确定配置目录路径
    if [ "$EUID" -eq 0 ]; then
        CONFIG_DIR="/root/gonka/deploy/join"
    else
        CONFIG_DIR="$HOME/gonka/deploy/join"
    fi
    
    # 检查配置目录是否存在
    if [ ! -d "$CONFIG_DIR" ]; then
        echo -e "${RED}[ERROR] Configuration directory not found: $CONFIG_DIR${NC}"
        echo -e "${YELLOW}[INFO] Please execute Command 1: Deploy Environment first${NC}"
        echo ""
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # 检查Configuration file是否存在
    CONFIG_FILE="$CONFIG_DIR/config.env"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}[ERROR] Configuration file not found: $CONFIG_FILE${NC}"
        echo -e "${YELLOW}[INFO] Please execute Command 3: Configure Environment Variables first${NC}"
        echo ""
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # Enter configuration directory
    cd "$CONFIG_DIR" || {
        echo -e "${RED}[ERROR] Cannot enter configuration directory${NC}"
        read -p "Press Enter to return to main menu..."
        return 1
    }
    
    # 检查 docker 是否可用
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}[ERROR] Docker not installed${NC}"
        echo -e "${YELLOW}[INFO] Please execute Command 1: Deploy Environment first${NC}"
        echo ""
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # 检查 docker-compose.yml file是否存在
    if [ ! -f "docker-compose.yml" ]; then
        echo -e "${RED}[ERROR] docker-compose.yml file not found${NC}"
        echo ""
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # 检查 docker-compose.mlnode.yml file是否存在
    if [ ! -f "docker-compose.mlnode.yml" ]; then
        echo -e "${YELLOW}[WARNING] docker-compose.mlnode.yml file not found${NC}"
        echo -e "${YELLOW}[INFO] Will only use docker-compose.yml${NC}"
        COMPOSE_FILES="-f docker-compose.yml"
    else
        COMPOSE_FILES="-f docker-compose.yml -f docker-compose.mlnode.yml"
    fi
    
    # Use sudo or run directly, depends on whether user is in docker group or is root
    DOCKER_COMPOSE_CMD="docker compose"
    if [ "$EUID" -ne 0 ] && ! groups | grep -q docker; then
        DOCKER_COMPOSE_CMD="sudo docker compose"
    fi
    
    # Load configuration并启动全节点
    echo -e "${GREEN}[STEP] Loading configuration file...${NC}"
    if source config.env; then
        echo -e "${GREEN}✓ Configuration file loaded successfully${NC}"
    else
        echo -e "${RED}[ERROR] Configuration file load failed${NC}"
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    echo ""
    
    # Modify BEACON_STATE_URL in docker-compose.yml file
    echo -e "${GREEN}[Step] Modifying docker-compose.yml configuration...${NC}"
    DOCKER_COMPOSE_FILE="docker-compose.yml"
    if [ -f "$DOCKER_COMPOSE_FILE" ]; then
        # Check if modification is needed (supports various formats: with quotes, without quotes, with spaces, etc.)
        if grep -q "BEACON_STATE_URL.*beaconstate\.ethstaker\.cc" "$DOCKER_COMPOSE_FILE"; then
            # Use sed for replacement (create temporary file then replace, safer)
            # Match various possible formats: https://beaconstate.ethstaker.cc/ or https://beaconstate.ethstaker.cc
            TEMP_FILE=$(mktemp)
            if sed -E 's|(BEACON_STATE_URL=.*)https://beaconstate\.ethstaker\.cc/?|\1https://beaconstate.info/|g' "$DOCKER_COMPOSE_FILE" > "$TEMP_FILE" && mv "$TEMP_FILE" "$DOCKER_COMPOSE_FILE"; then
                echo -e "${GREEN}✓ BEACON_STATE_URL updated to https://beaconstate.info/${NC}"
            else
                rm -f "$TEMP_FILE"
                echo -e "${YELLOW}[Warning] BEACON_STATE_URL update failed, continuing...${NC}"
            fi
        else
            echo -e "${YELLOW}[Info] BEACON_STATE_URL configuration not found or already updated${NC}"
        fi
    else
        echo -e "${YELLOW}[Warning] docker-compose.yml file not found${NC}"
    fi
    echo ""
    
    echo -e "${GREEN}[Step] Starting full node...${NC}"
    echo -e "${YELLOW}[Tip] This may take some time, please be patient...${NC}"
    echo ""
    
    if $DOCKER_COMPOSE_CMD $COMPOSE_FILES up -d; then
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}   Full Node Started Successfully!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        
        # Display service status
        echo -e "${CYAN}Service Status:${NC}"
        $DOCKER_COMPOSE_CMD $COMPOSE_FILES ps
        echo ""
        
        echo -e "${YELLOW}[Tip] You can use the following commands to check service status:${NC}"
        echo -e "${GREEN}   docker compose ps${NC}"
        echo -e "${GREEN}   docker compose logs -f${NC}"
        echo ""
    else
        echo ""
        echo -e "${RED}[Error] Full node startup failed${NC}"
        echo -e "${YELLOW}[Tip] Please check error message and retry${NC}"
        echo ""
    fi
    
    # Return to original directory
    cd "$ORIGINAL_DIR"
    
    echo ""
    read -p "Press Enter to return to main menu..."
}

# Command 7: Verify Node Status
command7_verify_node_status() {
    clear
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Command 7: Verify Node Status${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    # Save current working directory
    ORIGINAL_DIR=$(pwd)
    
    # 确定配置目录路径
    if [ "$EUID" -eq 0 ]; then
        CONFIG_DIR="/root/gonka/deploy/join"
    else
        CONFIG_DIR="$HOME/gonka/deploy/join"
    fi
    
    # 检查配置目录是否存在
    if [ ! -d "$CONFIG_DIR" ]; then
        echo -e "${RED}[ERROR] Configuration directory not found: $CONFIG_DIR${NC}"
        echo -e "${YELLOW}[INFO] Please execute Command 1: Deploy Environment first${NC}"
        echo ""
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # 检查Configuration file是否存在
    CONFIG_FILE="$CONFIG_DIR/config.env"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}[ERROR] Configuration file not found: $CONFIG_FILE${NC}"
        echo -e "${YELLOW}[INFO] Please execute Command 3: Configure Environment Variables first${NC}"
        echo ""
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    
    # Enter configuration directory
    cd "$CONFIG_DIR" || {
        echo -e "${RED}[ERROR] Cannot enter configuration directory${NC}"
        read -p "Press Enter to return to main menu..."
        return 1
    }
    
    # Load configuration
    echo -e "${GREEN}[STEP] Loading configuration file...${NC}"
    if source config.env; then
        echo -e "${GREEN}✓ Configuration file loaded successfully${NC}"
    else
        echo -e "${RED}[ERROR] Configuration file load failed${NC}"
        cd "$ORIGINAL_DIR"
        read -p "Press Enter to return to main menu..."
        return 1
    fi
    echo ""
    
    # Get user input wallet address
    echo -e "${CYAN}Please enter your gonka cold wallet address:${NC}"
    echo -e "${YELLOW}   This is the wallet address you created in Command 2${NC}"
    echo ""
    while [ -z "$GONKA_COLD_ADDRESS" ]; do
        read -p "Wallet address: " GONKA_COLD_ADDRESS
        if [ -z "$GONKA_COLD_ADDRESS" ]; then
            echo -e "${RED}   [Error] Wallet address cannot be empty, please re-enter${NC}"
        fi
    done
    echo -e "${GREEN}   ✓ Wallet address: $GONKA_COLD_ADDRESS${NC}"
    echo ""
    
    # Determine SEED API URL
    if [ ! -z "$SEED_API_URL" ]; then
        SEED_API_BASE="$SEED_API_URL"
    else
        SEED_API_BASE="http://node2.gonka.ai:8000"
    fi
    
    # Generate verification address
    VERIFY_URL="$SEED_API_BASE/v1/participants/$GONKA_COLD_ADDRESS"
    
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Node Status Verification Address${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    echo -e "${GREEN}Please copy the following address to your browser to verify node status:${NC}"
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}$VERIFY_URL${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${CYAN}Usage Instructions:${NC}"
    echo -e "${YELLOW}  1. Copy the above address${NC}"
    echo -e "${YELLOW}  2. Open the address in a browser${NC}"
    echo -e "${YELLOW}  3. View node status information${NC}"
    echo ""
    
    # Try to verify using curl (if available)
    if command -v curl &> /dev/null; then
        echo -e "${GREEN}[Step] Attempting to verify node status...${NC}"
        echo ""
        if curl -s -f "$VERIFY_URL" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Node status verification address accessible${NC}"
            echo ""
            echo -e "${CYAN}Node Status Information:${NC}"
            curl -s "$VERIFY_URL" | head -n20 || echo -e "${YELLOW}   Cannot parse response content${NC}"
            echo ""
        else
            echo -e "${YELLOW}[Warning] Cannot access verification address${NC}"
            echo -e "${YELLOW}[Tip] Please check:${NC}"
            echo -e "${YELLOW}   1. Node services are started (execute Command 6)${NC}"
            echo -e "${YELLOW}   2. Network connection is normal${NC}"
            echo -e "${YELLOW}   3. Wallet address is correct${NC}"
            echo ""
        fi
    else
        echo -e "${YELLOW}[Tip] curl not installed, cannot verify automatically${NC}"
        echo -e "${YELLOW}[Tip] Please manually open the above address in browser for verification${NC}"
        echo ""
    fi
    
    # Return to original directory
    cd "$ORIGINAL_DIR"
    
    echo ""
    read -p "Press Enter to return to main menu..."
}

# 显示主菜单
show_main_menu() {
    clear
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}   Gonka Deployment Management Script (Ubuntu)${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    echo -e "${GREEN}Twitter: @ferdie_jhovie${NC}"
    echo -e "${RED}Do not trust paid scripts, this script is free and open source${NC}"
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo ""
    echo -e "${GREEN}Please select an operation:${NC}"
    echo ""
    echo -e "${YELLOW}  1. Deploy Environment${NC}"
    echo -e "${YELLOW}      - Install Docker${NC}"
    echo -e "${YELLOW}      - Configure GPU support${NC}"
    echo -e "${YELLOW}      - Install HuggingFace CLI${NC}"
    echo -e "${YELLOW}      - Download deployment files${NC}"
    echo -e "${YELLOW}      - Install inferenced${NC}"
    echo ""
    echo -e "${YELLOW}  2. Create Wallet${NC}"
    echo -e "${YELLOW}      - Create gonka account key${NC}"
    echo ""
    echo -e "${YELLOW}  3. Configure Environment Variables${NC}"
    echo -e "${YELLOW}      - Configure node name, password, ports, etc.${NC}"
    echo ""
    echo -e "${YELLOW}  4. Check Sync Status${NC}"
    echo -e "${YELLOW}      - View tmkms and node service logs${NC}"
    echo ""
    echo -e "${YELLOW}  5. Create ML Operation Key${NC}"
    echo -e "${YELLOW}      - Create ML operation key in API container${NC}"
    echo ""
    echo -e "${YELLOW}  6. Start Full Node${NC}"
    echo -e "${YELLOW}      - Start all Docker services${NC}"
    echo ""
    echo -e "${YELLOW}  7. Verify Node Status${NC}"
    echo -e "${YELLOW}      - Generate verification address and check node status${NC}"
    echo ""
    echo -e "${YELLOW}  0. Exit${NC}"
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

# Main function
main() {
    while true; do
        show_main_menu
        read -p "Please enter option [0-7]: " choice
        echo ""
        
        case $choice in
            1)
                command1_deploy_environment
                ;;
            2)
                command2_create_wallet
                ;;
            3)
                command3_configure_env
                ;;
            4)
                command4_check_sync_status
                ;;
            5)
                command5_create_ml_key
                ;;
            6)
                command6_start_full_node
                ;;
            7)
                command7_verify_node_status
                ;;
            0)
                echo -e "${GREEN}Thank you for using, goodbye!${NC}"
                echo ""
                exit 0
                ;;
            *)
                echo -e "${RED}[ERROR] Invalid option, please re-select${NC}"
                echo ""
                read -p "Press Enter to continue..."
                ;;
        esac
    done
}

# Run main function
main
