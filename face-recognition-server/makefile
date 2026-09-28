DOCKER_IMAGE := face-recognition-server
DOCKER_USERNAME := tonhom260
TAG := latest
DOCKERFILE := Dockerfile
REGISTRY := docker.io
PLATFORM := linux/amd64  #
IMAGE_TAG := ghcr.io/$(DOCKER_USERNAME)/$(DOCKER_IMAGE):$(TAG)

buildx:
	@echo "Buildx the Docker image and push to repo... $(DOCKER_USERNAME)/$(DOCKER_IMAGE):$(TAG)"	
	docker buildx build --network host  --push --platform linux/amd64 -t ghcr.io/$(DOCKER_USERNAME)/$(DOCKER_IMAGE):$(TAG) .

dev:
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload
start:
	uvicorn main:app --host 0.0.0.0 --port 8000
