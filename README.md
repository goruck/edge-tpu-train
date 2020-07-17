# TensorFlow Object Detection API with Coral Edge TPU

This project uses the TensorFlow Object Detection API to train models suitable for the Google Coral Edge TPU. Follow the steps below to install the required programs and to train your own models for use on the Edge TPU.

## Installation

### Install TensorFlow Object Detection API
Follow these installation [steps](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html).

### Install edge-tpu-train
```bash
$ git clone https://github.com/goruck/models/tree/edge-tpu-train
```

Check the ```requirements.txt``` file to ensure you have the necessary Python packages installed on your system or virtual environment. 

### Setup Data Set Directories

Under the ```annotations```, ```images```, ```tf-record``` and ```tflite-models``` directories in ```edge-tpu-train``` place a sub-directory named after your data set(s).

### Install labelImg
Follow these [steps](https://github.com/tzutalin/labelImg) to install labelImg, a great tool you can use to label your images that you'll use for training. 

## Collect Images

Place your data set images in the ```images/<named-data-set>``` directory you created in the step above.

About 200 images per class is sufficient to re-train most models in my experience.

## Label Images

Use labelImg to label the images you collected. Store the xml annotation files in ```annotations/<named-data-set>```.

## Create Label Map (.pbtxt)

Classes need to be listed in the label map. Since in the case I am detecting the members of my family (including pets) the label map looks like this:

```protobuf
item {
    id: 1
    name: 'lindo'
}
item {
    id: 2
    name: 'nikki'
}
item {
    id: 3
    name: 'eva'
}
item {
    id: 4
    name: 'nico'
}
item {
    id: 5
    name: 'polly'
}
item {
    id: 6
    name: 'rebel'
}
item {
    id: 7
    name: 'unknown'
}
```

Note that id 0 is reserved. Store this file in the ```annotations/<named-data-set>``` folder with the name ```label_map.pbtxt```.

## Create TFRecord (.record)

TFRecord is an important data format designed for Tensorflow. (Read more about it [here](https://www.tensorflow.org/tutorials/load_data/tf_records)). Before you can train your custom object detector, you must convert your data into the TFRecord format.

Since you need to train as well as validate your model, the data set will be split into training (```train.record```) and validation sets (```val.record```). The purpose of training set is straightforward - it is the set of examples the model learns from. The validation set is a set of examples used DURING TRAINING to iteratively assess model accuracy.

Use the program [create_tf_record.py](./create_tf_record.py) to convert the data set into train.record and val.record.

This program is preconfigured to do 80–20 train-val split. Execute it by running:

```bash
$ python3 ./create_tf_record.py --dataset_name <named-data-set>
```

As configured above the program will store the ``.record`` files to the ```tf_record/<named-data-set>``` folder. 

## Download pre-trained model

There are many pre-trained object detection models available in the model zoo but you need to limit your selection to those that can be converted to quantized TensorFlow Lite (object detection) models. (You must use [quantization-aware training](https://www.tensorflow.org/model_optimization/guide/quantization/training), so the model must be designed with fake quantization nodes.)

In order to train them using your custom data set, the models need to be restored in Tensorflow using their checkpoints (```.ckpt``` files), which are records of previous model states.

For this example download ```ssd_mobilenet_v2_quantized_coco``` from [here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz) and save its model checkpoint files (```model.ckpt.meta```, ```model.ckpt.index```, ```model.ckpt.data-00000-of-00001```) to the ```checkpoints``` directory.

## Modify Config (.config) File

If required (for example you are changing the number of classes from 7 used in this example to something else) modify the files in the ```config/<named-data-set>``` directory as needed. There should not be many changes required if using the scripts above as directed except for the name of your data set. 

## Re-train model

Follow the steps below to re-train the model replacing the values for ```pipline_config_path``` and ```num_training_steps``` as needed. I found 1400 training steps to be sufficient in this example. 

```bash
$ train.sh \
--pipeline_config_path ./configs/pipeline_mobilenet_v2_ssd_retrain_last_few_layers.config \
--num_training_steps 1400
```

## Monitor training progress

Start tensorboard in a new terminal:

```bash
$ tensorboard --logdir ./train
```

## Convert model to TF Lite and compile it for the edge tpu

Run the following script to export the model to a frozen graph, convert it to a TF Lite model and compile it to run on the edge TPU. Replace the pipeline configuration path as required and make sure the checkpoint number matches the last training step used in training the model.

NB: this assumes the [Edge TPU Compiler](https://coral.withgoogle.com/docs/edgetpu/compiler/) has been installed on your system.

```bash
$ convert_checkpoint_to_edgetpu.sh \
--pipeline_config_path ./configs/pipeline_mobilenet_v2_ssd_retrain_last_few_layers.config \
--checkpoint_num 1400
```

## Run the model

You can now use the retrained and compiled model with the [Edge TPU Python API](https://coral.withgoogle.com/docs/edgetpu/api-intro/).

## License

[MIT](./LICENSE)