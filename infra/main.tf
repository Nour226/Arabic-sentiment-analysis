terraform {
  required_version = ">= 1.6.0"
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {}

resource "docker_network" "mlops" {
  name = "arabic-sentiment-mlops"
}

resource "docker_volume" "minio" {
  name = "arabic-sentiment-minio"
}

resource "docker_container" "minio" {
  name  = "arabic-sentiment-minio"
  image = "bitnamilegacy/minio:latest"

  command = ["server", "/bitnami/minio/data", "--console-address", ":9001"]

  env = [
    "MINIO_ROOT_USER=${var.minio_root_user}",
    "MINIO_ROOT_PASSWORD=${var.minio_root_password}",
  ]

  networks_advanced {
    name = docker_network.mlops.name
  }

  ports {
    internal = 9000
    external = 9000
  }

  ports {
    internal = 9001
    external = 9001
  }

  volumes {
    volume_name    = docker_volume.minio.name
    container_path = "/bitnami/minio/data"
  }
}
