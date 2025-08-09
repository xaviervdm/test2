# test2
This is test 2

## Semantic Segmentation Model

The `segmentation_model.py` script defines a simple U-Net built with `tf.keras` for performing semantic segmentation on 256x256 RGB images.

### Training

The `train_unet` function in `segmentation_model.py` compiles and fits the model. Provide `tf.data.Dataset` objects that yield `(image, mask)` pairs:

```python
from segmentation_model import train_unet

train_unet(train_ds, val_ds, epochs=10)
```
=======