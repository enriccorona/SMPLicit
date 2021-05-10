SMPLicit: Topology-aware Generative Model for Clothed People
=======

## Fitting SMPLicit

To fit SMPLicit, you must have installed it as a library as mentioned in the upper folder's README. Fitting requires SMPL estimations and image cloth segmentations. This release will fit the clothes of all people in the image, so we also take advantage of semantic segmentation to remove noise and occluded body parts. 

In summary, we built upon the following models:

- SMPL estimation: [FrankMoCap](https://github.com/facebookresearch/frankmocap). This is an amazing project which works great and is easy to use. You can use other SMPL estimation models and save predicted pose and shape on a .pkl file.
- Cloth segmentation and instance segmentation: [RP-RCNN](https://github.com/soeaver/RP-R-CNN). This work is from ECCV 2020 and will directly produce a cloth semantic segmentation and instance segmentation of all people in the image. There might be other models such as [this](https://github.com/PeikeLi/Self-Correction-Human-Parsing.git) which will be easier to run for this task too.

We provide a sample of the data used for testing under the `data/` folder and you can try with your own data by placing it under those folders or by changing the path in `options/image_fitting_options.py`. For running and recovering videos on the sample images, run the command

```
python fit_SMPLicit.py
```
