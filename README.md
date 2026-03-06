
------------------------
## Data
Download the validation set from: https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/
Rename the metadata to `train_metadata`

-----------------------
## Setup and Training

Just download `uv`: https://docs.astral.sh/uv/ and then run `uv sync`. 

To train the base model (for now) just run: 

```bash
python src/contrail-segmentation/train/trainer.py [ARGS]
```