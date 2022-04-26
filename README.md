# OrthoIrCam

## Compilation
If not alreday installed, install compilation tools. In Ubuntu for example,
```
sudo apt-get install build-essential
```
For matplotlib plotting you also need some latex packages.
```
sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
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
test data are in `data_test/`
the whole process of orthorectification need to run in order:

1. `driver.py` is Algo1
1. `refine_lwir.py` is Algo2
1. `ssim_prev4.py` is the filtering
1. `plot_lwir_finalSelection.py`
1. `get_ignition_line.py`

Follow description of each set:

### Algo1 from the manuscript
using the test case
```
run driver.py -i 1 -m lwir -s False
```
to get flag description 
```
run driver.py -h
```
`-i 1` is to run the test case


### Algo2 fromt the manuscript
not set up for python3 yet

### Filtering and ploting  
not set up for python3 yet


