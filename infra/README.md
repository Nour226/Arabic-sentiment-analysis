# Reproducible local artifact storage

This Terraform configuration creates the MinIO artifact-store network and container used by the local MLflow workflow.

```bash
terraform init
terraform apply -auto-approve
terraform destroy -auto-approve
```

Use `terraform apply -var='minio_root_password=<strong-password>'` for shared environments. The Docker Compose stack remains the recommended development entry point because it also starts MLflow and the API.
