# mphys-project
Understanding uncertainties with machine learning for radio astronomy

Training a LeNetKG model on image data will output the prediction logit and a measure of uncertainty $s$. The uncertainty on the logit can be calculated using

$$
\sigma^2 = \textrm{softplus}(s)
$$


