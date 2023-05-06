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
