provider "aws" {
  region = "ca-central-1"
}

module "key_pair" {
  source   = "./modules/key_pair"
  key_name = "nearflow-key"
  pub_key  = var.PROD_SSH_PUB_KEY
}

module "security_group" {
  source      = "./modules/security_group"
  name        = "nearflow-sg"
  description = "Allow access to Nearflow and system services"
  vpc_id      = "vpc-d3a346bb"
}

resource "aws_ebs_volume" "nearflow_volume" {
  availability_zone = "ca-central-1a"
  size              = 20
  type              = "gp3"
  tags = {
    Name = "nearflow-storage"
  }
}

resource "aws_instance" "nearflow_instance" {
  ami                    = "ami-00a8f8ec53d00a658"
  instance_type          = "t3.large"
  key_name               = module.key_pair.key_name
  vpc_security_group_ids = [module.security_group.id]
  availability_zone      = "ca-central-1a"
  user_data              = templatefile("${path.module}/nearflow-cloud-init.yaml", {
    POSTGRES_USER                        = var.POSTGRES_USER,
    POSTGRES_PASSWORD                    = var.POSTGRES_PASSWORD,
    POSTGRES_DB                          = var.POSTGRES_DB,
    LANGFLOW_SUPERUSER                   = var.LANGFLOW_SUPERUSER,
    LANGFLOW_SUPERUSER_PASSWORD          = var.LANGFLOW_SUPERUSER_PASSWORD,
    BACKEND_URL                          = var.BACKEND_URL,
    LANGFLOW_AUTO_LOGIN                  = var.LANGFLOW_AUTO_LOGIN,
    LANGFLOW_CACHE_TYPE                  = var.LANGFLOW_CACHE_TYPE,
    LANGFLOW_CONFIG_DIR                  = var.LANGFLOW_CONFIG_DIR,
    LANGFLOW_DATABASE_URL                = var.LANGFLOW_DATABASE_URL,
    LANGFLOW_ENV                         = var.LANGFLOW_ENV,
    LANGFLOW_HOST                        = var.LANGFLOW_HOST,
    LANGFLOW_LANGCHAIN_CACHE             = var.LANGFLOW_LANGCHAIN_CACHE,
    LANGFLOW_LOG_FILE                    = var.LANGFLOW_LOG_FILE,
    LANGFLOW_LOG_LEVEL                   = var.LANGFLOW_LOG_LEVEL,
    LANGFLOW_OPEN_BROWSER                = var.LANGFLOW_OPEN_BROWSER,
    LANGFLOW_PORT                        = var.LANGFLOW_PORT,
    LANGFLOW_REDIS_CACHE_EXPIRE          = var.LANGFLOW_REDIS_CACHE_EXPIRE,
    LANGFLOW_REDIS_DB                    = var.LANGFLOW_REDIS_DB,
    LANGFLOW_REDIS_HOST                  = var.LANGFLOW_REDIS_HOST,
    LANGFLOW_REDIS_PORT                  = var.LANGFLOW_REDIS_PORT,
    LANGFLOW_REMOVE_API_KEYS             = var.LANGFLOW_REMOVE_API_KEYS,
    LANGFLOW_SAVE_DB_IN_CONFIG_DIR       = var.LANGFLOW_SAVE_DB_IN_CONFIG_DIR,
    LANGFLOW_STORE_ENVIRONMENT_VARIABLES = var.LANGFLOW_STORE_ENVIRONMENT_VARIABLES,
    LANGFLOW_WORKERS                     = var.LANGFLOW_WORKERS,
    GHCR_PAT                             = var.GHCR_PAT,
    PROD_SSH_HOST                        = var.PROD_SSH_HOST,
    PROD_SSH_KEY                         = var.PROD_SSH_KEY,
    PROD_SSH_PUB_KEY                     = var.PROD_SSH_PUB_KEY,
  })

  tags = {
    Name = "Nearflow-Prod"
  }

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
  }

  depends_on = [aws_ebs_volume.nearflow_volume]
}

resource "aws_volume_attachment" "nearflow_attachment" {
  device_name = "/dev/sdf"
  volume_id   = aws_ebs_volume.nearflow_volume.id
  instance_id = aws_instance.nearflow_instance.id
  force_detach = true
}

resource "aws_eip" "nearflow_ip" {
  instance   = aws_instance.nearflow_instance.id
  domain     = "vpc"
  depends_on = [aws_instance.nearflow_instance]
}

data "aws_route53_zone" "vitalpoint" {
  name         = "vitalpoint.ai."
  private_zone = false
}

resource "aws_route53_record" "nearflow_dns" {
  zone_id = data.aws_route53_zone.vitalpoint.zone_id
  name    = "ai.vitalpoint.ai"
  type    = "A"
  ttl     = 300
  records = [aws_eip.nearflow_ip.public_ip]
}

# ⚠️ COMMENTED OUT: IAM role must be bootstrapped manually to avoid circular dependency

# data "aws_caller_identity" "current" {}

# resource "aws_iam_role" "nearflow_github_oidc_role" {
#   name = "nearflow-deploy-role"
#   assume_role_policy = jsonencode({
#     Version = "2012-10-17",
#     Statement = [
#       {
#         Effect = "Allow",
#         Principal = {
#           Federated = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/token.actions.githubusercontent.com"
#         },
#         Action = "sts:AssumeRoleWithWebIdentity",
#         Condition = {
#           StringLike = {
#             "token.actions.githubusercontent.com:sub" = "repo:ALuhning/nearflow:*"
#           },
#           StringEquals = {
#             "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
#           }
#         }
#       }
#     ]
#   })
# }

# resource "aws_iam_role_policy" "nearflow_deploy_policy" {
#   name = "nearflow-deploy-permissions"
#   role = aws_iam_role.nearflow_github_oidc_role.id

#   policy = jsonencode({
#     Version = "2012-10-17",
#     Statement = [
#       {
#         Effect   = "Allow",
#         Action   = [
#           "ec2:*",
#           "route53:*",
#           "iam:PassRole",
#           "acm:*",
#           "elasticloadbalancing:*"
#         ],
#         Resource = "*"
#       }
#     ]
#   })
# }
