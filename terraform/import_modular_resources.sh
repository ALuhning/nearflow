
#!/bin/bash

echo "Importing pre-existing AWS resources into Terraform state..."

# Replace with actual values from AWS Console
SECURITY_GROUP_ID="sg-03667cf066a2333c0"
KEY_PAIR_NAME="nearflow-key"

# Import EC2 key pair into module path
terraform import module.key_pair.aws_key_pair.this $KEY_PAIR_NAME

# Import security group into module path
terraform import module.security_group.aws_security_group.this $SECURITY_GROUP_ID

echo "✅ Import complete. Make sure you verify terraform plan before apply."
