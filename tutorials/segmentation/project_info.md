

## Validation Set
As the test set contains no ground trouth we decided to created a fixed validation set of 10 samples. A 5-fold Cross Validation was the original idea but unfortunately we ha no time to run all the experiments


## Metrics/Losses plot during and after training
We used Tensorboard for log and plot all the losses.


## Augmentations
We applied augmentations using torchio:
Random Gaussian noise (mean 0 std 2): probability 0.5
RandomAffine Transformations (scales=0.1,degrees=5): probability 0.8


## Results

Results are taken at min validation Loss

| Experiment      | Epoch | Dice Score         | IoU                |
|-----------------|-------|--------------------|--------------------|
| DICE NO_AUG     | 958   | 0.8196353614330292 | 0.7030864953994751 |
| DICE + AUG      | 994   | 0.8241025686264039 | 0.7106675028800964 |
| SOFTDICE NO_AUG | 900   | 0.8212120831012726 | 0.7049741446971893 |
| SOFTDICE + AUG  | 750   | 0.8513024926185608 | 0.74664785861969   |