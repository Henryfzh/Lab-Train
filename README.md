# Usage
```
cd nanoGPT

python lightning_train.py
```

# Experiment Report: Optimizer Performance Comparison for Coin Flip Prediction

## Introduction

This experiment aimed to compare the computational efficiency of two popular optimizers, Stochastic Gradient Descent (SGD) and Adam, when applied to a simple neural network model trained on a coin flip dataset. The focus was on the time taken by the `optimizer_zero_grad` operation, which resets gradients before backpropagation.

## Methodology

### Dataset and Model

- **Dataset**: A custom dataset was created (`GenerateDataset`) where each sequence represents a series of coin flips with a probability of heads set to 0.666. The dataset includes 1000 samples with each sequence having a length of 10 flips.

- **Model**: A simplified version of a transformer model (`GPT`) with nanoGPT was used. The model only contains 1 layer for fast training.

### Optimizer Comparison

- **SGD**: Known for its simplicity, SGD updates parameters based on the gradient of the loss function for each mini-batch.
- **Adam**: An adaptive learning rate method that computes individual learning rates for different parameters from estimates of the first and second moments of the gradients.

### Training Procedure

- The model was trained for one epoch using both SGD and Adam optimizers with identical learning rates.
- The time taken for the `optimizer_zero_grad` operation was measured for each optimizer over all iterations.

### Data Analysis

- **Timing Data**: The time taken by `optimizer_zero_grad` was recorded for each optimizer during training.
- **Bootstrap Resampling**: To estimate the distribution of mean times, bootstrap resampling was performed 1000 times for each optimizer's timing data.

### Visualization

- **Box Plot**: Showcased the distribution of times for each optimizer, highlighting central tendency and variability.
- **Bootstrap Distribution**: A histogram was created to visualize the distribution of mean times from bootstrap resampling, allowing for comparison between SGD and Adam.

## Results

- **Box Plot**: 
  ![Boxplot](nanoGPT/boxplot.png "Boxplot Results")
  The box plot indicated that:
  - SGD generally showed a tighter distribution of times, suggesting less variability in its `optimizer_zero_grad` operation.
  - Adam exhibited a broader distribution, potentially due to its additional computations for adaptive learning rates.

- **Bootstrap Analysis**: 
  ![Bootstrap Mean Distributions](nanoGPT/bootstrap.png "Bootstrap Results")
  The bootstrap distributions showed:
    - SGD's mean time distribution was more concentrated around its mean, indicating more consistent performance.
    - Adam's distribution was wider, showing higher variability in mean times.

- **Time**
    ```
    sgd_times: [0.00018596649169921875, 8.845329284667969e-05, 8.678436279296875e-05, 4.673004150390625e-05, 3.8623809814453125e-05, 4.315376281738281e-05, 4.38690185546875e-05, 3.719329833984375e-05, 4.1961669921875e-05, 4.1961669921875e-05, 4.3392181396484375e-05, 4.2438507080078125e-05, 3.647804260253906e-05, 4.267692565917969e-05, 3.647804260253906e-05, 3.5762786865234375e-05, 3.552436828613281e-05, 3.528594970703125e-05, 3.504753112792969e-05, 4.267692565917969e-05, 4.57763671875e-05, 4.458427429199219e-05, 3.647804260253906e-05, 3.600120544433594e-05, 3.528594970703125e-05, 3.62396240234375e-05, 3.62396240234375e-05, 4.291534423828125e-05, 3.5762786865234375e-05, 4.220008850097656e-05, 3.5762786865234375e-05, 3.62396240234375e-05]
    
    adam_times: [8.463859558105469e-05, 6.556510925292969e-05, 5.14984130859375e-05, 4.9591064453125e-05, 5.626678466796875e-05, 9.489059448242188e-05, 4.76837158203125e-05, 4.267692565917969e-05, 4.363059997558594e-05, 4.2438507080078125e-05, 5.221366882324219e-05, 4.100799560546875e-05, 4.0531158447265625e-05, 3.719329833984375e-05, 3.7670135498046875e-05, 4.5299530029296875e-05, 4.6253204345703125e-05, 5.125999450683594e-05, 3.8623809814453125e-05, 3.6716461181640625e-05, 4.1484832763671875e-05, 8.58306884765625e-05, 5.555152893066406e-05, 3.7670135498046875e-05, 3.719329833984375e-05, 3.695487976074219e-05, 4.553794860839844e-05, 4.458427429199219e-05, 4.4345855712890625e-05, 4.267692565917969e-05, 3.981590270996094e-05, 4.506111145019531e-05]
    ```

## Conclusion

- **SGD** proved to be faster per iteration due to its simplicity, as expected. However, its total training time might be longer if more iterations are needed for convergence.
- **Adam**, while slower per iteration, might require fewer iterations to reach a good solution due to its adaptive learning rate, potentially making it more efficient in practice.
- The variability in Adam's timing suggests that in scenarios where computational resources are limited, SGD might be preferred for its consistency, whereas Adam could be chosen for its ability to adapt learning rates for better convergence.
