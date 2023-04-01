SMPLicit: Topology-aware Generative Model for Clothed People
=======

[[Project]](http://www.iri.upc.edu/people/ecorona/smplicit/) [[arXiv]](https://arxiv.org/abs/2103.06871)<!-- TODO: Fitting SMPLicit -->

<img src='http://www.iri.upc.edu/people/ecorona/smplicit/teaser.png' width=800>

## License

Software Copyright License for non-commercial scientific research purposes. Please read carefully the terms and conditions and any accompanying documentation before you download and/or use the SMPLicit model, data and software, (the "Model & Software"), including 3D meshes, blend weights, blend shapes, textures, software, scripts, and animations. By downloading and/or using the Model & Software (including downloading, cloning, installing, and any other use of this github repository), you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Model & Software. Any infringement of the terms of this agreement will automatically terminate your rights under this License.

## Installation

Follow these commands to install SMPLicit in your environment. The required libraries are standard, with the possible exception of Kaolin which requires a particular version to run with the current code. 

- `git clone https://github.com/enriccorona/SMPLicit`
- `cd SMPLicit`
- Install the dependencies listed in [requirements.txt](requirements.txt):
  - pip install -r requirements.txt
- In particular, we use Kaolin v0.1 (see [installation command](https://kaolin.readthedocs.io/en/v0.1/notes/installation.html)) which should be easy to install. However, if you want to use a later version, you might need to update the import to TriangleMesh in `SMPLicit/SMPLicit.py`

- Download the SMPL model from [here](https://drive.google.com/file/d/1HsUW8jPfU6kHowRDHcURdwXskFbIIMet/view?usp=sharing) and place it in SMPLicit/utils/


To be able to import and use `SMPLicit` in another project, just use run `python setup.py install` in the main folder.

## Usage 

To check that everything is going well, run one of the test scripts under the examples folder. The first example will just show a simple T-Shirt on a standard shaped SMPL and visualize it using trimesh, to make sure everything is working.

```
cd examples/
python example.py
```

SMPLicit can represent clothes of different types, so the following example will also add lower-body clothes, hair and shoes into the example:

```
python example_fullbody.py
```

And finally one can interpolate between clothes of different types. For instance, moving between a jacket, tops, short or long sleeved T-Shirts. The following script will generate object meshes that represent these clothes and will be saved in interpolation/, below the main folder.

```
python interpolate.py
```

## Fitting SMPLicit from a single image:

We used another script that builds on SMPL estimation to find the cloth types that best fit the cloth semantic segmentation. Please refer to the README under `Fit_SMPLicit/`.

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
