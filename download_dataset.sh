#!/usr/bin/env bash

cd data

wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

tar -xvf annotations.tar.gz
tar -xvf images.tar.gz