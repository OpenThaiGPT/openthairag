# Gunicorn configuration file

# Bind address and port
bind = "0.0.0.0:5000"

# Number of worker processes
workers = 1

# Number of threads per worker
threads = 1

# Worker class
worker_class = "sync"

# Timeout for worker processes
timeout = 120

# Maximum number of requests a worker will process before restarting
max_requests = 1000

# Maximum number of simultaneous clients
worker_connections = 1000

# Access log format
accesslog = "-"

# Error log format
errorlog = "-"

# Log level
loglevel = "info"

# Preload application code before worker processes are forked
preload_app = True

# Reload workers when code changes (only for development)
reload = False

# SSL configuration (uncomment and modify if using HTTPS)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"

# Set environment variables
raw_env = [
    "FLASK_ENV=production",
    "PYTHONUNBUFFERED=1",
    "LOG_LEVEL=INFO"
]

capture_output = True
enable_stdio_inheritance = True