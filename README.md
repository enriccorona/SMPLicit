SMPLicit: Topology-aware Generative Model for Clothed People
=======

[[Project]](http://www.iri.upc.edu/people/ecorona/smplicit/) [[arXiv]](https://arxiv.org/abs/2103.06871)<!-- TODO: Fitting SMPLicit -->

## License

Software Copyright License for non-commercial scientific research purposes. Please read carefully the terms and conditions and any accompanying documentation before you download and/or use the SMPLicit model, data and software, (the "Model & Software"), including 3D meshes, blend weights, blend shapes, textures, software, scripts, and animations. By downloading and/or using the Model & Software (including downloading, cloning, installing, and any other use of this github repository), you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Model & Software. Any infringement of the terms of this agreement will automatically terminate your rights under this License.

## Installation

- `git clone https://github.com/ecorona/SMPLicit`
- `cd SMPLicit`
- Install the dependencies listed in [requirements.txt](requirements.txt)
  - pip install -r requirements.txt
- In particular, we use Kaolin v0.1 (see [installation command](https://kaolin.readthedocs.io/en/v0.1/notes/installation.html)) which should be easy to install. However, if you want to use a later version, you might need to update the import to TriangleMesh)

- Download the SMPL model from [here](https://drive.google.com/file/d/19plO4du6uXv8beTtEo0K3iYIyHu1YRHu/view?usp=sharing) and place it in SMPLicit/utils/

## Install SMPLicit package

To be able to import and use `ManoLayer` in another project, go to your `manopth` folder and run `pip install .`


`cd /path/to/other/project`

## Usage 

To check that everything is going well, run one of the test scripts under the examples folder.

```
cd examples/
python example.py # This will just show a simple T-Shirt on a standard shape SMPL and visualize it, to make sure everything is well set up
python example_fullbody.py # This example will generate different cloth types on SMPL
python interpolate.py # This example will generate object meshes that interpolate on upper-body clothes. These are saved in the interpolation folder
```

## Citation

If you find the code useful, please cite: 

```
@inproceedings{corona2021smplicit,
    Author = {Enric Corona and Albert Pumarola and Guillem Aleny{\`a} and Pons-Moll, Gerard and Moreno-Noguer, Francesc},
    Title = {SMPLicit: Topology-aware Generative Model for Clothed People},
    Year = {2021},
    booktitle = {CVPR},
}
```
