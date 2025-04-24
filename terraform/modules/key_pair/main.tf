
variable "key_name" {
  type        = string
  description = "Name of the EC2 key pair"
}

variable "pub_key" {
  type        = string
  description = "Public SSH key for the key pair"
}

resource "aws_key_pair" "this" {
  key_name   = var.key_name
  public_key = var.pub_key

  lifecycle {
    ignore_changes  = [public_key]
    prevent_destroy = true
  }
}

output "key_name" {
  value = aws_key_pair.this.key_name
}
