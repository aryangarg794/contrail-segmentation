
------------------------
## Data
The data downloading is much simpler now: just make sure to add your kaggle api key ([linux](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md)) or json ([windows](https://www.kaggle.com/discussions/getting-started/524433), use the legacy api thing). Then just run: 

```bash
uv run download-data
```

-----------------------
## Setup and Training

Just download `uv`: https://docs.astral.sh/uv/ and then run `uv sync`. 

To train the base model (for now) just run: 

```bash
uv run train 
```

The project uses hydra-core so you can define your models as a yaml file: https://hydra.cc/docs/intro/. Basically define the python class that you want for your model (lightning module) then, define a yaml file and you can just select the 
model as a cli argument like:

```bash
uv run train model=your_model 
```

where your yaml file looks like this: 

```yaml
_target_: contrail_segmentation.models.pretrained_unet.PretrainedUNET

lr: 1e-3 
wd: 1e-4
beta1: 0.9
beta2: 0.999
threshold: 0.5

encoder_class:
  _partial_: True
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet50
  encoder_weights: ssl
  in_channels: 24
  classes: 1

```

The target just points to your target python class. The rest are arguments for the class used in initialization. The encoder class is a *partial* which means that it clears a python obj (thats callable) with the given parameters.This means in the code you can call `encoder_class()` and it will make a class of `segmentation_models_pytorch.Unet` with the given parameters.  

With hydra you can also pass other args in a similar manner. I setup a data file with batch size, mask only, augmentations etc, and you can just change those variables inside the CLI instead of code like this: 
```bash 
uv run train data.batch_size=16 model.lr=1e-5 
```

This also saves your config to the wandb run, making it easier to identify runs and backtrack if you need to. 