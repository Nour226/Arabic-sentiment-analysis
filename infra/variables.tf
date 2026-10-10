variable "minio_root_user" {
  type      = string
  sensitive = true
  default   = "minioadmin"
}

variable "minio_root_password" {
  type      = string
  sensitive = true
  default   = "minioadmin123"
}
