# Small children privacy protection through automatic face covering by emojis in photos

This project constitutes the final project for the course "Deep Neural Networks" (2020-2021).

## Create the dataset

Create a directory and insert some images containing faces in there. Use the full path of that directory in the `PATH_TO_DATASET` variable inside `src/utils/face_detection_yolo.py`.

```bash
$ cd src/utils
$ python3 face_detection_yolo.py
```

By default, the extracted faces will be stored in `src/utils/faces_extracted` but this can be changed by modifying the `OUTPUT_PATH` variable.

Finally, manually choose what faces belong to infants and what does not, the first ones should be added to a directory `babies` and the second ones to a directory named `not-babies`.

## Train

## Perform predictions

In order to perform predictions with an already trained model, enter the `src/predict` directory. Set the `PATH_TO_DEMO_IMGS` and `PATH_TO_EMOJIS` inside the `perform_predictions.py`. The first directory corresponds to the directory that the images for the demo will be stored and the second one is the directory that contains the available images that will be used to hide the faces of infants. Set also the `PATH_TO_MODEL` variable to point to the `model.h5` file of the trained model.

```bash
$ python3 perform_predictions.py
```