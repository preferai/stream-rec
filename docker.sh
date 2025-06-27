#!/bin/bash
# Docker management script for HOMETOWN recommendation API

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running!"
        log_info "Please start Docker Desktop or your Docker daemon first"
        log_info "On macOS with Colima: colima start"
        log_info "On macOS with Docker Desktop: Open Docker Desktop app"
        exit 1
    fi
}

# Check if data directory exists
check_data() {
    if [ ! -d "data" ] || [ ! -f "data/users.parquet" ] || [ ! -f "data/streams.parquet" ]; then
        log_error "Data directory or required files missing!"
        log_info "Please ensure data/users.parquet and data/streams.parquet exist"
        exit 1
    fi
}

# Build the Docker image
build() {
    check_docker
    log_info "Building HOMETOWN API Docker image..."
    docker-compose build
    log_success "Build complete!"
}

# Start services in production mode
start() {
    check_docker
    check_data
    log_info "Starting HOMETOWN API services..."
    docker-compose up -d
    log_success "Services started!"
    log_info "API available at: http://localhost:8000"
    log_info "API docs at: http://localhost:8000/docs"
}

# Start services in development mode
dev() {
    check_docker
    check_data
    log_info "Starting HOMETOWN API in development mode..."
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
}

# Stop services
stop() {
    log_info "Stopping HOMETOWN API services..."
    docker-compose down
    log_success "Services stopped!"
}

# View logs
logs() {
    docker-compose logs -f hometown-api
}

# Test the API
test() {
    log_info "Testing HOMETOWN API..."
    if ! curl -s http://localhost:8000/ >/dev/null; then
        log_error "API is not responding. Make sure it's running with: $0 start"
        exit 1
    fi
    
    # Run the test script inside the container
    docker-compose exec hometown-api python test_api.py
}

# Clean up everything
clean() {
    log_warning "This will remove all containers, images, and volumes..."
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose down -v
        docker image rm stream-rec_hometown-api 2>/dev/null || true
        log_success "Cleanup complete!"
    else
        log_info "Cleanup cancelled"
    fi
}

# Show help
help() {
    echo "🏠 HOMETOWN Recommendation API - Docker Management"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build    Build the Docker image"
    echo "  start    Start services in production mode"
    echo "  dev      Start services in development mode (with live reload)"
    echo "  stop     Stop all services"
    echo "  logs     Show API logs"
    echo "  test     Test the API endpoints"
    echo "  clean    Remove all containers and images"
    echo "  help     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build && $0 start    # Build and start"
    echo "  $0 dev                  # Development mode"
    echo "  $0 logs                 # View logs"
}

# Main command handling
case "${1:-help}" in
    build)
        build
        ;;
    start)
        start
        ;;
    dev)
        dev
        ;;
    stop)
        stop
        ;;
    logs)
        logs
        ;;
    test)
        test
        ;;
    clean)
        clean
        ;;
    help|*)
        help
        ;;
esac
