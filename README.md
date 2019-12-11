# cat_faces

Pets segmentation with deep neural networks.

## Net

This project uses ResNetLW and MobileNetLW from [here](https://github.com/DrSleep/light-weight-refinenet).

## Dataset

The Oxford-IIIT Pet Dataset from [here](https://www.robots.ox.ac.uk/~vgg/data/pets).

## Howto:

* make development environment using docker:

```bash
$ python run_dev.py
```

* download dataset_data:

```bash
$ ./download_dataset.sh
```

* train net:

```bash
python src/train_model.py
```