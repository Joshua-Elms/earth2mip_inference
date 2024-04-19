## Introduction

Author: [Joshua Elms](jmelms@iu.edu)

This repository is a version of earth2mip-main that was forked in March 2024 to add simplified tools for running inference. The inference code in this guide only supports the [FourCastNetV2 Small Model (fcnv2_sm)](https://doi.org/10.48550/arXiv.2306.03838), but it could be adapted to work with the other models from [earth2mip](https://github.com/NVIDIA/earth2mip). 

Two primary modes are provided for inference, depending on where the initialization conditions are sourced from. If the user wishes to run inference on ECWMF Reanalysis 5 ([ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)) conditions, the "Simple" inference mode takes only a date/time as input and will download the necessary data from the European Center for Medium-Range Weather Forecast's ([ECMWF](https://www.ecmwf.int/)) Climate Data Store ([CDS](https://cds.climate.copernicus.eu/#!/home)). If the user needs to run inference on non-ERA5 conditions, they can use the "Custom" inference mode. This requires the user to format data in compliance with the `earth2mip.initial_conditions.hdf5` class, details on which are provided in `models.fcnv2_sm.inference_utils`.

This guide covers both the initial setup of the repository and methods to perform "Simple" and "Custom" inference runs of fcnv2_sm. 

## Setup

1. Set your working directory to `earth2mip_inference`. Download model weights from [modulus](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/modulus_fcnv2_sm). Because the model weights are ~3 GB, it's recommended to keep them on a high-capacity system. This may not be your home directory. For Indiana University users, the Slate filesystem is the best choice. Once you've moved the model weights to the desired location, unzip them. Symlink the unzipped model weights from wherever you've placed them to the `models/fcnv2_sm/` directory, making sure to use absolute paths when creating the symlink.
    ```
    cd earth2mip_inference

    wget 'https://api.ngc.nvidia.com/v2/models/nvidia/modulus/modulus_fcnv2_sm/versions/v0.2/files/fcnv2_sm.zip'

    mv fcnv2_sm.zip <storage_dir>/

    unzip <storage_dir>/fcnv2_sm.zip -d <storage_dir>

    rm <storage_dir>/fcnv2_sm.zip

    ln -s <abs_path_to_storage_dir>/fcnv2_sm/weights.tar models/fcnv2_sm/
    ```

2. Activate a conda environment that has all necessary libraries for earth2mip. For users in Travis O'Brien's lab group, the following command can be used:
    ```
    conda activate /N/project/obrienta_startup/wxmod_ai/conda/earth2mip
    ```

3. Set the model registry environment variable. This must be run every time the shell is restarted, so if the user plans to run inference regularly, this line can be added to their `~/.bashrc` file to set the variable on startup.
    ```
    export MODEL_REGISTRY="<abs_path_to_earth2mip_parent>/earth2mip_inference/models"
    ```

## Inference

All of the tools provided by this repository to assist with inference are located in `models/fcnv2_sm`. The below steps assume that is the working directory. 

### Simple Mode (ERA5 input)

1. Set necessary parameters in `run_inference.py`. The initial conditions for inference are decided exclusively by the `start_time` parameter. The data will be pulled from ERA5, which means that it can range from 01/01/1940 00:00:00 to a few days before the present day.

2. Either call the python inference script directly, or submit a SLURM job for it. The provided `run_inference.sh` script is an example for Indiana University HPC users. If using that script, modify the `source activate <env>` line accordingly to use the `earth2mip` environment.


### Custom Mode (Any NetCDF input)

The steps for custom inference are same as the above, with the exception that the `start_time` parameter does not affect the dataset when calling the function `run_custom_inference`. Rather, the `data_source_path` should set to point to a valid data source. An example of a properly structured data source directory is given at `example_custom_data/`, but the `example_custom_data/data/1970.h5` file is empty and will not produce valid inference runs. Details on formatting your data for use with fcnv2_sm are provided in the documentation of the function `inference_utils.py:run_custom_inference`.  

### Notes
1. If using the Simple inference mode, the initial conditions might be cached on your system for future use. 
2. With the exception of code in `models/`, no modifications of code in this repository will take effect when calling functions from the `earth2mip` module, unless you have built your own conda environment and run the commands to build `earth2mip` from this repository in that environment.
3. Unlike earlier versions of FCN, fcnv2_sm performs all standardization of input and output data itself; custom data should be kept at its actual scale. 
4. fcnv2_sm can be run on cpu and gpu, but the author of this guide has not tested this code on gpu. It should perform inference more quickly and have no other effects.