#!/bin/bash
# Starts Weaviate with all required environment variables.
# Data is stored at /home/weaviate_data which is persistent on Azure App Service.

export QUERY_DEFAULTS_LIMIT=25
export AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
export PERSISTENCE_DATA_PATH=/home/weaviate_data
export DEFAULT_VECTORIZER_MODULE=none
export ENABLE_MODULES=text2vec-openai,generative-openai
export CLUSTER_HOSTNAME=node1

mkdir -p /home/weaviate_data

exec /usr/local/bin/weaviate \
    --host 0.0.0.0 \
    --port 8080 \
    --scheme http
