## Prerequisites for running ODL CT

The following two methods can be used to install the dependencies for running the ODL CT demo. We use Conda for the installation of all packages.

### Method 1: Set up the environment using the YAML file.
 For the full set of required Python packages, create a Conda environment from the provided YAML.
```
conda create -f odl_env.yml
```

### Method 2: Set up the environment manually.
If you would like to install the dependencies without using the shared environment, following the instructions step by step to install them manually. A detailed instruction reference can be found [here](https://github.com/odlgroup/odl).

Create a conda environment

```
conda create --name odl_env python=3.8
```

Activate the environment

```
conda activate odl_env
```

Install the following packages one by one in the created environment:
```
conda install numpy
conda install -c pytorch pytorch
conda install scipy
pip install https://github.com/odlgroup/odl/archive/master.zip
conda install -c astra-toolbox astra-toolbox
conda install -c conda-forge matplotlib
```

## Run the ODL CT demo
In the ODL environment created above, run
```
python odl_demo.py
```

