# FLED for Radar Beampattern Design under Constant Modulus Constraint


## Description

FLED is a deep learning algorithm that employs the concept of algorithm unrolling to solve the beampattern design problem and the constant modulus constraint.


## Architecture Details

Although the code is pretty flexible and can be configured the way you may like, we use the following setup in our expirements.


1. The network structure is visualized as follows.

![Architecture](imgs/fled.png)

where the projection and retraction operations are defined as follows.

![project_and_retract](imgs/project_retract_visualize.png)

and the exansion operation concatenates to the its inputs as follows.

![expand](imgs/expand_visualization.png)


2. The implementation details of each unrolled step (block in FLED) is defined as follows.

![details](imgs/unrolled_step.png)


3. The initialization block is defined as follows.

![initialize](imgs/initialize.png)


4. The direction evaluation module is defined as follows.

![direction](imgs/direction.png)


5. The architecture details of the expansion layer is as follows.

![expand](imgs/expand.png)


6. We set the number of unrolled steps $\kappa$ to $5$.


## Code 

### Train

A train example is in [train.sh](FLED/train.sh). You can either use it or play with the configurations to your linking such as follows.
```sh
cd FLED
python train.py \
  --model-config "model1" \
  --train-config "train1" \
  --dataset-config "dataset_random" \
  --loss-config "loss1" \
  --model "Model1" \
  --datamodule "BeamPatternDataModule" \
  --callbacks "LogValBeampatternResult" \
              "LogValNumericalResult" \
              "LogScenarioResult"
```

all configs are set in [configs](FLED/configs) where we have separate folders for [models](FLED/configs/models/), [train](FLED/configs/train/), [losses](FLED/configs/losses/) and [datasets](FLED/configs/datasets/)

### Test


An example for testing the code is found in [test.py](FLED/test.py). You can use it as follows.
```sh
python test.py \
  --model-config "model1" \
  --train-config "train1" \
  --dataset-config "dataset_random" \
  --loss-config "loss_grad_guide" \
  --model "Model1" \
  --datamodule "BeamPatternDataModule" \
  --callbacks "LogValBeampatternResult" "LogValNumericalResult" "LogScenarioResult" \
  --file "path_to_the_checkpoint_file.ckpt"
```


### Structure

1. [utils](FLED/utils): contains some utilities that are used throughout the code.
Most importantly, it contains [base_classes.py](FLED/utils/base_classes.py) where it has all the configuration classes.

2. [configs](FLED/configs): contains all examples of configurations used for datasets + training models + testing.

3. [dataset.py](FLED/datasets/base_dataset.py): contains the dataset module to load data that was saved from Matlab and prepares it for PyTorch model.


