
variable "POSTGRES_USER" {
  description = "Postgres username"
  type        = string
}

variable "POSTGRES_PASSWORD" {
  description = "Postgres password"
  type        = string
  sensitive   = true
}

variable "POSTGRES_DB" {
  description = "Postgres database name"
  type        = string
}

variable "LANGFLOW_SUPERUSER" {
  description = "Langflow superuser username"
  type        = string
}

variable "LANGFLOW_SUPERUSER_PASSWORD" {
  description = "Langflow superuser password"
  type        = string
  sensitive   = true
}

variable "BACKEND_URL" {
  description = "Langflow backend url"
  type        = string
  sensitive   = true
}

variable "LANGFLOW_AUTO_LOGIN" {
  description = "Langflow auto login"
  type        = string
  sensitive   = true
}

variable "LANGFLOW_CACHE_TYPE" {
  description = "Langflow cache type"
  type        = string
  sensitive   = true
}

variable "LANGFLOW_CONFIG_DIR" {
  description = "Langflow config dir"
  type        = string
  sensitive   = true
}

variable "LANGFLOW_DATABASE_URL" {
  description = "Langflow database url"
  type        = string
  sensitive   = true
}

variable "LANGFLOW_ENV" {
  description = "Langflow env"
  type        = string
  sensitive   = true
}

variable "LANGFLOW_HOST" {
  description = "Langflow host"
  type        = string
  sensitive   = true
}

variable "LANGFLOW_LANGCHAIN_CACHE" {
  description = "Langflow langchain cache"
  type        = string
  sensitive   = true
}
          
variable "LANGFLOW_LOG_LEVEL" {
  description = "Langflow log level"
  type        = string
  sensitive   = true
}

variable "LANGFLOW_LOG_FILE" {
  description = "Langflow log file"
  type        = string
  sensitive   = true
}

variable "LANGFLOW_OPEN_BROWSER" {
  description = "Langflow open browser"
  type        = string
  sensitive   = true
}

variable "LANGFLOW_PORT" {
  description = "Langflow port"
  type        = string
  sensitive   = true
}

variable "LANGFLOW_REDIS_CACHE_EXPIRE" {
  description = "Langflow redis cache expire"
  type        = string
  sensitive   = true
}

variable "LANGFLOW_REDIS_HOST" {
  description = "Langflow redis host"
  type        = string
  sensitive   = true
}
          
variable "LANGFLOW_REDIS_PORT" {
  description = "Langflow redis port"
  type        = string
  sensitive   = true
}

variable "LANGFLOW_REDIS_DB" {
  description = "Langflow redis db"
  type        = string
  sensitive   = true
}

variable "LANGFLOW_REMOVE_API_KEYS" {
  description = "Langflow remove api keys"
  type        = string
  sensitive   = true
}

variable "LANGFLOW_SAVE_DB_IN_CONFIG_DIR" {
  description = "Langflow save db in config dir"
  type        = string
  sensitive   = true
}

variable "LANGFLOW_STORE_ENVIRONMENT_VARIABLES" {
  description = "Langflow store environment variables"
  type        = string
  sensitive   = true
}

variable "LANGFLOW_WORKERS" {
  description = "Langflow workers"
  type        = string
  sensitive   = true
}

variable "PROD_SSH_PUB_KEY" {
  description = "Public key for EC2 key pair"
  type        = string
}

variable "PROD_SSH_HOST" {
  description = "host for EC2 key pair"
  type        = string
}

variable "PROD_SSH_KEY" {
  description = "Private key for EC2 key pair"
  type        = string
}

variable "GHCR_PAT" {
  type        = string
  description = "GitHub Container Registry Personal Access Token"
  sensitive   = true
}