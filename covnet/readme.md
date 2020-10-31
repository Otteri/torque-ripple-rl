#  Convolutional neaural network based compensator
To train the compensator, run `python train.py --steps 20`
For better visualizations, you can train inverted model: `python train.py --steps 20 --invert`
With inverted model, the train output should look like the input. Visualizations can be found from predictions `folder/`

Convolution based model is trained with reduced pulsation model. (Model only generates harmonics, but it does not model inertia and such physical phenomena). Data is generated automatically.

Example command:  
`python train.py --step 15 --invert`
