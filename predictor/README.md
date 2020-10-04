# Predictor
This is an independent minimodule for learning periodical waveforms. Model must be trained with input data, which can be generated e.g. running `generate_data.py`. Generated data can be first viewed with `plot_data.py` before starting training with `train.py`. Training progress can be occasionally checked by viewing plots in `predictions/` folder.

---

## Config parameters explained
`Config.py` file provides many tweaking options. In order to avoid cluttering the config file, the parameters are explained here. Currently, there are three main section in the config: *Model*, *Data generation* and *Plotting*. Parameters of each section is described below.

###      Model
`Learning_rate:` how fast the model should try to learn. Step size for the optimizer.  
`Hidden_layers:` amount of neurons in hidden layers. Greater amount allows model to learn more complicated signals but learning speed slows down simultaneously.



### Data generation
The data used for training the model can look following:
```
[[0.1, 0.2, ..., 0.9]
 [0.3, 0.4, ..., 1.2]
 [0.5, 0.6, ..., 1.4]
 [1.0, 1.1, ..., 1.9]]
```
Each row-vector is a single training set, a sample of a signal. Matrix has configurable dimensions N x L, where  
`L:` signal length  
`N:` number of signals  

The generated signal sample can consist of n-periods. When generating the signal, the number of periods in a sample can be adjusted setting `repetitions` parameters. The generated data is saved to location that `datafile` parameter specifies.

#### Signal shape
First number is the harmonic order which has relation to frequency,
and second number gives the amplitude of the n-th harmonic.
Specifying only single harmonic results to plain sine wave.
```
harmonics = {
    1 : 0.5,
}
```

More advanced signal. Note that not every order has to be given.
```
harmonics = {
    1 : 0.02,
    2 : 0.0030,
    6 : 0.006,
    12: 0.00227,
    24: 0.00039
}
````


### Plotting
`color:` Color (check available colors from matplotlib document)  
`dpi:` Output image resolution
