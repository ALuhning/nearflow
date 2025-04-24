# Get current account ID dynamically
data "aws_caller_identity" "current" {}

# GitHub OIDC Identity Provider
resource "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"

  client_id_list = ["sts.amazonaws.com"]

  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831e3780aea1"
  ]
}

# Role for GitHub Actions to assume via OIDC
resource "aws_iam_role" "nearflow_github_oidc_role" {
  name = "nearflow-deploy-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Federated = aws_iam_openid_connect_provider.github.arn
        },
        Action = "sts:AssumeRoleWithWebIdentity",
        Condition = {
          StringLike = {
            "token.actions.githubusercontent.com:sub" = "repo:ALuhning/nearflow:*"
          },
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })
}

# Permissions for deploy actions (you can scope down later)
resource "aws_iam_role_policy" "nearflow_deploy_permissions" {
  name = "nearflow-deploy-policy"
  role = aws_iam_role.nearflow_github_oidc_role.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = [
          "ec2:*",
          "iam:PassRole",
          "route53:*",
          "acm:*",
          "elasticloadbalancing:*"
        ],
        Resource = "*"
      }
    ]
  })
}
