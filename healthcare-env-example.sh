# Healthcare Research Data Pipeline Environment Configuration
# Copy this file to .env and update with your values

# Database Configuration
DB_USER=healthcare_admin
DB_PASSWORD=your_secure_password_here
DB_NAME=healthcare_research
DB_HOST=localhost
DB_PORT=5432

# For Docker Compose
DATABASE_URL=postgresql://healthcare_admin:your_secure_password_here@postgres:5432/healthcare_research

# Redis Configuration
REDIS_PASSWORD=your_redis_password_here
REDIS_URL=redis://default:your_redis_password_here@redis:6379/0

# Security Keys (Generate new ones for production!)
SECRET_KEY=your-secret-key-here-generate-with-fernet
JWT_SECRET=your-jwt-secret-key-here-minimum-32-chars

# Epic Integration (Production values)
EPIC_CLARITY_CONNECTION=mssql+pymssql://username:password@epic-server:1433/CLARITY
EPIC_CABOODLE_CONNECTION=mssql+pymssql://username:password@epic-server:1433/CABOODLE
EPIC_API_KEY=your-epic-api-key
EPIC_API_SECRET=your-epic-api-secret
EPIC_BASE_URL=https://your-epic-instance.com/api

# FHIR Server Configuration
FHIR_SERVER_URL=https://your-fhir-server.com/fhir
FHIR_AUTH_TOKEN=your-fhir-auth-token

# Application Settings
LOG_LEVEL=INFO
APP_ENV=development
DEBUG=False
WORKERS=4

# API Configuration
API_HOST=0.0.0.0
API_PORT=8001
API_RATE_LIMIT=100
API_RATE_LIMIT_PERIOD=3600

# Admin Configuration
PGADMIN_EMAIL=admin@healthcare.com
PGADMIN_PASSWORD=your_pgadmin_password_here

# Monitoring
GRAFANA_PASSWORD=your_grafana_password_here
PROMETHEUS_RETENTION_DAYS=30

# Email Configuration (for alerts)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL=alerts@healthcare.com

# Data Retention Settings
DATA_RETENTION_DAYS=2555  # 7 years for HIPAA
AUDIT_LOG_RETENTION_DAYS=2190  # 6 years for HIPAA

# Performance Settings
MAX_BATCH_SIZE=10000
PARALLEL_WORKERS=4
QUERY_TIMEOUT_SECONDS=300

# Encryption Settings
ENCRYPTION_ALGORITHM=AES256
KEY_ROTATION_DAYS=90

# Feature Flags
ENABLE_REAL_TIME_ANALYTICS=true
ENABLE_PREDICTIVE_MODELS=false
ENABLE_AUDIT_LOGGING=true
ENABLE_DATA_MASKING=true

# External Services
AZURE_STORAGE_CONNECTION=DefaultEndpointsProtocol=https;AccountName=...
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1

# SSL/TLS Configuration
SSL_CERT_PATH=/etc/ssl/certs/healthcare.crt
SSL_KEY_PATH=/etc/ssl/private/healthcare.key
FORCE_HTTPS=true

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_SCHEDULE=0 2 * * *  # 2 AM daily
BACKUP_RETENTION_DAYS=30
BACKUP_LOCATION=/backups

# Compliance Settings
HIPAA_MODE=true
GDPR_MODE=false
AUDIT_MODE=strict