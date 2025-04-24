
terraform {
  backend "s3" {
    bucket         = "vitalpoint-nearflow-terraform-state"  # Replace if bucket already exists
    key            = "prod/terraform.tfstate"
    region         = "ca-central-1"
    encrypt        = true
  }
}
