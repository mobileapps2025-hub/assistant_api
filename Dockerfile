# Stage 1: extract the Weaviate binary from the official image
FROM semitechnologies/weaviate:1.27.0 AS weaviate-stage

# Stage 2: main application image
FROM python:3.11-slim

WORKDIR /app

# Install system deps: ODBC for SQL Server, supervisor, wget (healthcheck)
RUN apt-get update && apt-get install -y \
    unixodbc \
    unixodbc-dev \
    curl \
    gnupg2 \
    supervisor \
    wget \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl https://packages.microsoft.com/config/debian/12/prod.list | tee /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

# Copy Weaviate binary from its official image
COPY --from=weaviate-stage /bin/weaviate /usr/local/bin/weaviate

# Copy startup scripts
COPY scripts/start-weaviate.sh /scripts/start-weaviate.sh
COPY scripts/start-api.sh /scripts/start-api.sh
RUN chmod +x /scripts/start-weaviate.sh /scripts/start-api.sh

# Copy supervisord config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create log directory
RUN mkdir -p /var/log/supervisor

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Azure App Service exposes port 8000; Weaviate runs on 8080 (internal only)
EXPOSE 8000

ENV MODULE_NAME="app.main"
ENV VARIABLE_NAME="app"
ENV PORT="8000"
# Point the app at the local Weaviate instance
ENV WEAVIATE_URL="http://localhost:8080"

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
