#!/bin/bash

echo "Importing pre-existing AWS resources into Terraform state..."

# Replace with actual values from AWS Console
SECURITY_GROUP_ID="sg-03667cf066a2333c0"
KEY_PAIR_NAME="nearflow-key"

# Run from Terraform directory, explicitly loading the tfvars file
terraform import -var-file=terraform.tfvars module.key_pair.aws_key_pair.this $KEY_PAIR_NAME
terraform import -var-file=terraform.tfvars module.security_group.aws_security_group.this $SECURITY_GROUP_ID

echo "✅ Import complete. You can now run terraform plan or let GitHub Actions take over."