# OrthoIrCam


## Funding
<img src="data_static/img/msca.jpg"
     alt="MSCA"
     style="float: right; margin-left: 1rem; display: block; max-width: 50px" />
This code was developped within the [3DFireLab](https://3dfirelab.eu/) project, a project funded by the European Union’s Horizon 2020 research and innovation program under the Marie Skłodowska-Curie agreement, grant H2020-MSCA-IF-2019-892463. 


## Compilation
If not alreday installed, install compilation tools. In Ubuntu for example,
```
sudo apt-get install build-essential
```
For matplotlib plotting you also need some latex packages.
```
sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super texlive-math-extra
```
Then, install an anaconda environment with the libraray listed in the yml file you can find [here](https://www.dropbox.com/s/b7j0iwsqd7295rh/AnacondaEnvMypy3Moritz.yml?dl=0)

If you have question handling anaconda env see [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment)

Download a modified version of opencv based on version 3.4.3 with some homebrewed modidication for the ECC function.
This is quite old now, to compile it I had to disable manny opencv package that are of no use for the current work.
the opencv code source is available [here](https://www.dropbox.com/s/3ta70bhjm1zyw2u/Opencv_343.tar.gz?dl=0)

There is instruction for compilation in the dir `Opencv_343/opencv` of the tar file. see file `compile_ronan.txt`
Your anaconda env should be loaded during the compilation.
Once compiled you need to link the python opencv librairy with your anaconda env.
```
ln -fs $WhereYouUnTarTheOpencV/Opencv_343/Lib/lib/python3.9/site-packages/cv2.cpython-39-x86_64-linux-gnu.so \
       $YourAnacondaEnvDir/lib/python3.9/site-packages
```
Note that if you are not using the same config as in the above yml file, you might have to modify the python version in the linking.


## Quick Description of the code
code are in `src/`
camera info are stored in `data_static/`
input congifuration file are in `input_config/`
The whole process of orthorectification need to run in this order:

1. `driver.py` is Algo1 
1. `refine_lwir.py` is Algo2 
1. `ssim_prev4.py` is the filtering 
1. `plot_final_lwir.py`

the three first steps are described in [Paugam et al 2021](https://doi.org/10.3390/rs13234913).

### Test Case 
`Ngarkat` is the data set provided with the code. 
It can be downloaded on the repository [dataverse.csuc.cat](https://doi.org/10.34810/data565)

the 'root' directory define in the config file should look like this.
Extracting the tar file form the above repository creates the Data directory and all sub directory.
```
.
├── Data
│   ├── FLIR570
│   │   ├── MAT
│   │   │   ├── > input data: frame_xxxxx.MAT
...
│   └── ignition_time.dat
└── Postproc
    ├── DEM
    │   ├── corrected_terrain_simpleHomography.png
    │   ├── Ngarkat_dem.txt
    │   ├── ngarkat_ngarkat_dem.npy
    │   ├── ngarkat_ngarkat_dem.png
    │   └── Ngarkat_plotE_polygon.kml
    ├── grid_ngarkat.npy
    ├── grid_ngarkat.prj
    ├── Instruments_Location
    │   ├── cornerFireNames.txt
    │   ├── Ngarkat_cf.txt
    │   ├── Ngarkat_gcps.kml
    │   └── Ngarkat_plotContour.kml
    ├── LWIR
    │   └── > output data processing 
    └── OrthoData
        └── > output final data
```
All files in Postproc directory are cretaed by the algorightms described bellow.
The two directorties 'Data/' and 'Postproc/' are named in the config file with variables 
`root_data` and `root_postproc`
Follows a quick description of each steps to run to perform the orthorectification.

### Algo1 from the manuscript: `driver.py`
A first algorigthm is aligning the images times series using on the first image that is manually georeference.
No fix ground control poins are required.
Using the test case
```
run driver.py -i Ngarkat -m lwir -s False
```
to get flag description 
```
run driver.py -h
```

### Algo2 from the manuscript: `refine_lwir.py`
A second algorithm loops again around the images time series focusing on area based alignement of the background scene.
```
run refine_lwir.py -i Ngarkat -s False
```

### Filtering: `ssim_prev4.py`
A last algorithm is applying filter to remove outilier images in the time series.
```
run ssim_prev4.py -i Ngarkat -s False
```

### plotting and saving
A python script is plotting LWIR frames in png format as well as 
creating a netcdf file with all frames.
```
run plot_final_lwir.py -i Ngarkat --angle 0
```
the `--angle` option is applying a rotation to the north-south orientated images in the png figure only if desired.

example of the LWIR frames times series for the Ngarkat fire is shown below:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=bIaeLFx3yBM 
" target="_blank"><img src="http://img.youtube.com/vi/bIaeLFx3yBM/0.jpg" 
alt="LWIR Ngarkat" width="480" height="360" border="10" /></a>
