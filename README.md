# [Tensorize, Factorize and Regularize: Robust Visual Relationship Learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hwang_Tensorize_Factorize_and_CVPR_2018_paper.pdf)

Seong Jae Hwang, Sathya N. Ravi, Zirui Tao, Hyunwoo J. Kim, Maxwell D. Collins, Vikas Singh, "Tensorize, Factorize and Regularize: Robust Visual Relationship Learning", Computer Vision and Pattern Recognition (CVPR), 2018.

## Dataset preparation
In total, there are files：
1. Images [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
2. [Image metadata](http://svl.stanford.edu/projects/scene-graph/VG/image_data.json)
3. [VG scene graph](http://svl.stanford.edu/projects/scene-graph/VG/VG-scene-graph.zip)

The image database file ```imdb_1024.h5``` is generated through files (1-3).

4. Scene graph database: [VG-SGG.h5](http://svl.stanford.edu/projects/scene-graph/dataset/VG-SGG.h5)
5. Scene graph database metadata: [VG-SGG-dicts.json](http://svl.stanford.edu/projects/scene-graph/dataset/VG-SGG-dicts.json)
6. RoI proposals: [proposals.h5](http://svl.stanford.edu/projects/scene-graph/dataset/proposals.h5)
7. RoI distribution: [bbox_distribution.npy](http://svl.stanford.edu/projects/scene-graph/dataset/bbox_distribution.npy)
8. [Faster-RCNN model](http://cvgl.stanford.edu/scene-graph/dataset/coco_vgg16_faster_rcnn_final.npy)

#### Case 1: Using preprocessed data (coming soon)
- [Model link](https://s3.amazonaws.com/tzr-tools/ckpt.tgz),  extract all files. In ```./checkpoints```, there are two files for baseline models```./checkpoints/Xu_2``` (by _Xu et al._), ```./checkpoints/CKP_Vrd``` (by _Lu et al._).

- Save [Faster-RCNN model](http://cvgl.stanford.edu/scene-graph/dataset/coco_vgg16_faster_rcnn_final.npy) to ```data/pretrained```.

- [Full imdb dataset, image metadata, VG Scene graph, ROI database and its metadata](https://s3.amazonaws.com/tzr-tools/data.tgz). Check that all files are under ```data/vg``` directory and contain following 5 files:
  ```
  imdb_1024.h5
  bbox_distribution.npy
  dproposals.h5
  VG-SGG-dicts.json
  VG-SGG.h5
  ```
- Download the VisualGenome [image_metadata](http://svl.stanford.edu/projects/scene-graph/VG/image_data.json) and its [scece_graph](http://svl.stanford.edu/projects/scene-graph/VG/VG-scene-graph.zip), extract the files and place all the jason files under ```./data_tools/VG```:
Check the following 3 files are under ```data_tools/VG``` directory:
  ```
  images_data.jason
  objects.jason
  relationships.jason
  ```


#### Case 2: Training from the scratch
You need the following 5 files:
1. Image database: ```imdb_1024.h5```
2. Scene graph database: [VG-SGG.h5](http://svl.stanford.edu/projects/scene-graph/dataset/VG-SGG.h5)
3. Scene graph database metadata: [VG-SGG-dicts.json](http://svl.stanford.edu/projects/scene-graph/dataset/VG-SGG-dicts.json)
4. RoI proposals: [proposals.h5](http://svl.stanford.edu/projects/scene-graph/dataset/proposals.h5)
5. RoI distribution: [bbox_distribution.npy](http://svl.stanford.edu/projects/scene-graph/dataset/bbox_distribution.npy)


(i). Download dataset images. [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

(ii). Save [Faster-RCNN model](http://cvgl.stanford.edu/scene-graph/dataset/coco_vgg16_faster_rcnn_final.npy) to ```data/pretrained```.
<!-- [//]: # ((iii）. Following the [Convert VisualGenome to desired format](https://github.com/danfeiX/scene-graph-TF-release/tree/master/data_tools#convert-visualgenome-to-desired-format) to generate five files under ```data/vg``` directory.)  -->

(iii). Place all the json files under ```data_tools/VG/```. Place the images under ```data_tools/VG/images```

(iii). Create image database file ```imdb_1024.h5``` by executing ```./create_imdb.sh``` in this directory. This script creates a hdf5 databse of images ```imdb_1024.h5```. The longer dimension of an image is resized to 1024 pixels and the shorter side is scaled accordingly. You may also create a image database of smaller dimension by editing the size argument of the script. You may skip to (vii) if you chose to downloaded (2-4).

(iv). Create an ROI database and its metadata by executing ```./create_roidb.sh``` in this directory. The scripts creates a scene graph database file ```VG-SGG.h5``` and its metadata ```VG-SGG-dicts.json```. By default, the script reads the dimensions of the images from the imdb file created in (iii). If your imdb file is of different size than 512 and 1024, you must add the size to the img_long_sizes list variable in the vg_to_roidb.py script.

(v). Use the script provided by py-faster-rcnn to generate (4)```proposal.h5```.

(vi). Change line 93 of tools/train_net.py to True to generate (5) ```bbox_distribution.npy```.

(vii). Finally, place (1-5) in ```data/vg```.

(viii). Check that all files are under ```data/vg``` directory and contain following 5 files:
  ```
  imdb_1024.h5
  bbox_distribution.npy
  dproposals.h5
  VG-SGG-dicts.json
  VG-SGG.h5
  ```

## Installing dependencies
required dependencies: 
- Python 2.7
- [TensorFlow r0.12](https://github.com/tzrtzr000/Tensorflow-GPU-install-instructions-for-lab-machine-ubuntu-14.04-and-16.04)
- [h5py](http://www.h5py.org/)
- [numpy 1.11.0](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [scipy 0.12.0](https://www.scipy.org/)
- [pyyaml](https://pypi.python.org/pypi/PyYAML)
- [easydict](https://pypi.python.org/pypi/easydict/)
- [cython](http://cython.org/)
- [Pillow 2.3.0](https://pillow.readthedocs.io/en/5.3.x/)
- [graphviz](https://pypi.python.org/pypi/graphviz) (optional, if you wish to visualize the graph structure)
- CUDA 8.0 

1. Create python 2.7 environment:
```
conda create -n tfr python=2.7
source activate tfr
```

2. Installing dependenciy packages: 
```
pip install -r requirement.txt
```
## Note: Make sure that your tensorflow version is __r.012__ GPU enabled version.
(helpful instruction [here](https://github.com/tzrtzr000/Tensorflow-GPU-install-instructions-for-lab-machine-ubuntu-14.04-and-16.04) for installing tensorflow r0.12 on ubuntu 14.04/16.04 and associated software supports).


## Compiling ROI pooling layer library
1. After you have installed all the dependencies, run the following command to compile nms and bbox libraries:
```
cd lib
make
```

2. Follow this [this instruction](https://github.com/danfeiX/scene-graph-TF-release/blob/master/lib/roi_pooling_layer) to see if you can use the pre-compiled roi-pooling custom op or have to compile the op by yourself. 


## Training (For Case 2 in Dataset preparation)
1.Run
```
./experiments/scripts/train.sh dual_graph_vrd_final 2 CHECKPOINT_DIRECTORY GPU_ID SIGMA
```
The program saves a checkpoint to ```./checkpoints/<_CHECKPOINT_DIRECTORY_>/``` every 50000 iterations. Training a full model on a desktop with Intel i7 CPU, 64GB memory, and a TitanX graphics card takes around 20 hours. You may use tensorboard to visualize the training process. By default, the tf log directory is set to ```checkpoints/<_CHECKPOINT_DIRECTORY_>/tf_logs/```.


## Evaluation 
1. Run
```
./experiments/scripts/test.sh <gpu_id> <checkpoint_dir> <checkpoint_file prefix> <model_options> <number_of_inference_for_dual_graph_vrd_fianl> <number_images> <mode>
```
Where <model_options> are:
```
 dual_graph_vrd_final by Xu et al (where our implementation is based on).

 vrd by Lu et al.
``` 
Three evaluation <mode> are:
```
sg_cls:  predict the predicated object and relationship (predicate) given the ground truth bounding boxes
sg_det (all): predicting object classification, relationship (predicate) prediction, using the proposed bounding box from the regional proposal network as object proposals
```
e.g.
```
/experiments/scripts/test.sh 0  CHECKPOINT_DIRECTORY FILE_PREFIX dual_graph_vrd_final 2 100 all
```

## Visualization
1. Run the same scripts in Evaluation: with one of the following three modes:
```
 viz_cls: visualize the sg_cls results
 viz_det: visualize the sg_det results
 viz_gt: visualizing the ground truth
```

-----------------
### Note: If the code is fetched from [__Xu et al.'s scene graph repository__](https://github.com/danfeiX/scene-graph-TF-release), then 
### It is imperative to change the following: 

1. __Change the line on 26 at *lib/roi_data_layer/minibatch.py*__ with the following code ([Learn more about why doing this here](https://github.com/rbgirshick/py-faster-rcnn/issues/481#issuecomment-337278950)): 
```
fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image).astype(np.int)
```

2. __Comment out code block from line 76 to 79 at [tools/test_net.py](https://github.com/danfeiX/scene-graph-TF-release/blob/master/tools/test_net.py), as checkpoints does not contain .ckpt files explicitly and tf.saver only needs correct file prefix to succesfully restore model__. Learn more about [here](https://stackoverflow.com/questions/41265035/tensorflow-why-there-are-3-files-after-saving-the-model). 

