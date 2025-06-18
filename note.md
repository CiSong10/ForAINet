# ForAINet Implement Note

The file names and directory structures referenced in this note correspond to my [branch](https://github.com/CiSong10/ForAINet/tree/ci). 
Please make adjustments accordingly if you're using a different setup.

## Set up  Environment

### Docker

```
# cd home/<username>/ForAINet
docker build -t for-ai-net ./setup
docker run -itd --gpus all -v home/<username>/ForAINet:home/<username>/ForAINet for-ai-net
docker exec -it <container_name> bash
```

### Mamba

I failed to build the environment using mamba
([PanopticSegForLargeScalePointCloud #17](https://github.com/prs-eth/PanopticSegForLargeScalePointCloud/issues/21)).
Commands in [appendix](#mamba-environment-setup-commands), to be fixed in the future.


## Training from Scratch

Download dataset: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10828636.svg)](https://doi.org/10.5281/zenodo.10828636)

Create a [Weights & Biases](https://wandb.ai/) account

**Modify configuration**

If we want to train using "basic setting", our command will be:

```
python3 PointCloudSegmentation/train.py \
 task=panoptic \
 data=panoptic/treeins_set1 \
 models=panoptic/FORpartseg_3heads \
 model_name=PointGroup-PAPER \
 training=treeins_set1 \
 job_name=basic
```

Notice the `training=treeins_set1` flag, that means we need to edit `conf/training/treeins_set1.yaml`.
In line 33, change wandb `entity` value to your wandb team name.
Adjust `epochs` (line 3). For me, one epoch takes around 8 minutes. 

Run the command above, or `bash commands/basic_setting.sh` if you are in my [branch](https://github.com/CiSong10/ForAINet/tree/ci) 

The results will be saved in `outputs/<job_name>/<job_name>-<model_name>-<date>_<time>/<model_name>.pt`

## Test

Modify configuration in `conf/eval.yaml` (the path to config is defined in `eval.py` line 7 `@hydra.main(config_path="conf", config_name="eval")` ):
1. `checkpoint_dir` to your model path
2. `data` is the (list of) paths for your test data

Hydro will load `config_path/config_name.yml` i.e. `conf/eval.yml` in this case.

```
python3 eval.py
```

Command for output the final evaluation file. Replace parameter `test_sem_path` by your path

```
python3 evaluation_stats_FOR.py
```

The evaluation results will be saved in `outputs/<model_path>/eval/<date_time>`

## Calculate Tree Metrics

Tree metrics calculation requires a different environment from the segmentation task. 
Please refer to [set up environment for tree metrics](#set-up-environment-for-tree-metrics) section for environment set up.

I refactored the `measurement.py` — use `measurement_new.py` for the updated modualized version. 
Adapt to your data in `main()` function. 
For instructions related to the original script, see the [Original measurement.py](#original-measurementpy) section.

If you're using `.laz` outputs from *ForestSens*, use `measurement_no_eval.py` to calculate tree metrics. 
This version is intended for segmentation outputs without corresponding ground truth data.

## Direct Deploy

(developing)

To directly deploy the model on your data, you need to structure it similarly to the provided sample data.
Your `.ply` data must include the following fields: `('x', 'y', 'z', 'intensity', 'semantic_seg', 'treeID')`
If your data does not include ground truth labels, you can assign dummy values (e.g., zeros) to `semantic_seg` and `treeID`. You can use the `prepare_data.py` script from my branch to help with this.

Edit the configuration file `PointCloudSegmentation/conf/predict.yaml` and run `PointCloudSegmentation/predict.py`.

## Finetune


## Appendix

### Mamba environment setup commands

This needs to be fixed in the future.

```
mamba create -n ex1 python=3.8.19
mamba activate ex1
mamba install pytorch=1.9.0=*cuda* torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

pip install numpy==1.19.5
mamba install openblas-devel -c anaconda

# Build MinkowskiEngine locally
cd MinkowskiEngine/
python setup.py install --blas=openblas 

pip install torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-geometric==1.7.2

pip install -r requirements.txt  # torch-cluster failed to build due to linux kernel mismatch

pip install numba==0.55.1
mamba install openblas-devel hdbscan -c anaconda -c conda-forge
mamba install anaconda::numpy-base==1.19.2

mamba install hydra-core
```

### Set up environment for tree metrics

GDAL (or one of its dependencies) was compiled against an older version of Intel TBB (`libtbb.so.2`), which was used in TBB 2020 or earlier. To ensure compatibility, we need to create a conda environment using an older version of TBB.

```
mamba create -n tree-metrics python=3.8 gdal pylidar tbb=2020.2 -c conda-forge -c rios
mamba activate tree-metrics
gdalinfo --version # Test if GDAL can run
mamba install -y matplotlib imageio scikit-image openpyxl plyfile alphashape numpy=1.23
mamba install -c conda-forge hdbscan tbb=2020.2
pip install python-opencv
pip install pyransac3d
```

### Original `measurement.py`

Adjust parameters in `metrics/measurement.py` line 28-40 `fold`, `file_path`, and line 635 `field_file_path`.

```
python tree_metrics/measurement.py
```
