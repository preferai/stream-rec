# Docker Setup for HOMETOWN Recommendation API

This document describes how to run the HOMETOWN recommendation API using Docker.

## Quick Start

1. **Build and start the services:**
   ```bash
   ./docker.sh build
   ./docker.sh start
   ```

2. **Test the API:**
   ```bash
   ./docker.sh test
   ```

3. **View the API documentation:**
   Open http://localhost:8000/docs in your browser

## Available Commands

Use the `docker.sh` script for easy management:

```bash
# Build the Docker image
./docker.sh build

# Start in production mode
./docker.sh start

# Start in development mode (with live reload)
./docker.sh dev

# Stop all services
./docker.sh stop

# View logs
./docker.sh logs

# Test the API
./docker.sh test

# Clean up everything
./docker.sh clean

# Show help
./docker.sh help
```

## Services

### HOMETOWN API (`hometown-api`)
- **Port:** 8000
- **Health Check:** http://localhost:8000/
- **API Docs:** http://localhost:8000/docs
- **Endpoints:**
  - `GET /` - Health check
  - `POST /v1/scenarios/hometown` - HOMETOWN recommendations
  - `GET /v1/scenarios/hometown/stats` - System statistics

### Redis (`hometown-redis`)
- **Port:** 6379
- **Purpose:** Caching (optional, for future enhancements)

## Data Requirements

The API requires the following data files in the `data/` directory:
- `users.parquet` - User profiles with location data
- `streams.parquet` - Stream metadata with creator locations

These files are mounted as read-only volumes into the container.

## Development Mode

For development with live code reloading:

```bash
./docker.sh dev
```

This mounts the source code and enables auto-reload when files change.

## Production Deployment

For production, use:

```bash
# Build the image
docker-compose build

# Start services in detached mode
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f hometown-api
```

## Environment Variables

The API supports these environment variables:

- `PYTHONUNBUFFERED=1` - Unbuffered Python output
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)

## Volumes

- `./data:/code/data:ro` - Synthetic dataset (read-only)
- `./logs:/code/logs` - Application logs
- `redis_data` - Redis persistence

## Health Checks

Both services include health checks:
- **API:** Checks if the service responds on port 8000
- **Redis:** Checks if Redis responds to ping

## Troubleshooting

### Service won't start
```bash
# Check logs
./docker.sh logs

# Check container status
docker-compose ps
```

### API returns 503 errors
This usually means the data files are missing or corrupted:
```bash
# Check if data files exist
ls -la data/

# Verify file contents
docker-compose exec hometown-api python -c "
import pandas as pd
print('Users:', len(pd.read_parquet('data/users.parquet')))
print('Streams:', len(pd.read_parquet('data/streams.parquet')))
"
```

### Port conflicts
If port 8000 is already in use:
```bash
# Check what's using the port
lsof -i :8000

# Or modify docker-compose.yml to use a different port
# Change "8000:8000" to "8001:8000"
```

## Testing the API

### Using the test script
```bash
./docker.sh test
```

### Manual testing with curl
```bash
# Health check
curl http://localhost:8000/

# HOMETOWN recommendation
curl -X POST http://localhost:8000/v1/scenarios/hometown \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_000001"}'

# System stats
curl http://localhost:8000/v1/scenarios/hometown/stats
```

### Using the interactive API docs
Visit http://localhost:8000/docs to test endpoints interactively.

## Monitoring

### Logs
```bash
# Follow logs
./docker.sh logs

# Or with docker-compose directly
docker-compose logs -f hometown-api
```

### Container stats
```bash
docker stats hometown-api hometown-redis
```

### Health status
```bash
docker-compose ps
```

## Cleanup

To remove everything (containers, images, volumes):
```bash
./docker.sh clean
```

This will prompt for confirmation before removing all Docker resources.
