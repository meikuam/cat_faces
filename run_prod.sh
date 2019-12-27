#!/usr/bin/env bash
cd docker
docker build -t cat_faces:prod prod
docker run -d -v $(readlink -e ../):/user/cat_faces --name=cat_faces-prod-instance-0 cat_faces:prod
