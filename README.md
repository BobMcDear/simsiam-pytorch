<i>This is not a fork of somebody else's code. I, [@BobMcDear](https://github.com/BobMcDear), am the original creator of this project but due to problems with Git was forced to delete and restore it. In other words, [gary109/PyTorch-SimSiam](https://github.com/gary109/PyTorch-SimSiam) is a fork of this repository and not vice versa.</i>

# SimSiam in PyTorch
This is an implementation of the self-supervised learning algorithm SimSiam in PyTorch.

# Usage
Many methods useful for SimSiam are included, and for end-to-end training, all you must provide is an encoder for the SimSiam model plus an optimizer.

Methods:

* ```data.py```

  * ```create_simsiam_transforms```: Returns a ```Compose``` object consisting of SimSiam's transforms and optionally normalization
    * Args:
      * ```size``` (```int```): Desired image size. Default is ```224```
      * ```normalize``` (```bool```): Whether to normalize with ImageNet's stats. Default is ```True```
  * ```SimSiamDataset```: Dataset for SimSiam that returns two augmented views per image with transforms from ```create_simsiam_transforms```
    * Args:
      * ```path``` (```str```): Path to images. Default is ```images/```
      * ```valid_exts``` (```List[str]```): Valid extensions for images. Default is ```['jpeg', 'jpg']```
      * ```size``` (```int```): Desired image size. Default is ```224```
      * ```normalize``` (```bool```): Whether to normalize with ImageNet's stats. Default is ```True```
  * ```create_simsiam_dataloader```: Wraps ```SimSiamDataset``` inside a ```DataLoader```
    * Args:
      * ```path``` (```str```): Path to images. Default is ```images/```
      * ```valid_exts``` (```List[str]```): Valid extensions for images. Default is ```['jpeg', 'jpg']```
      * ```size``` (```int```): Desired image size. Default is ```224```
      * ```normalize``` (```bool```): Whether to normalize with ImageNet's stats. Default is ```True```
      * ```batch_size``` (```int```): Batch size. Default is ```32```
      * ```num_workers``` (```int```): Number of workers. Default is ```8```


* ```loss.py```:
  * ```NegativeCosineSimilarity```: Negative cosine similarity
  * ```SymmetrizedNegativeCosineSimilarity```: Symmetrized negative cosine similarity loss from the SimSiam paper


* ```model.py```:
  * ```get_encoder_out_dim```: Returns the number of output channels of an encoder
    * Args:
      * ```encoder``` (```Module```): Encoder
  
  * ```LinBnReLU```: Linear layer followed optionally by batch normalization and ReLU
    * Args:
      * ```in_dim``` (```int```): Number of input features
      * ```out_dim``` (```int```): Number of output features
      * ```bn``` (```bool```): Whether to have batch normalization. Default is ```True```
      * ```relu``` (```bool```): Whether to have ReLU. Default is ```True```
  * ```SimSiamModel```: SimSiam model that uses a provided encoder and returns the ```z``` and ```p``` vectors
    * Args:
      * ```encoder``` (```Module```): Encoder
      * ```out_dim``` (```int```): Number of output features for projection and prediction heads. Default is ```2048```
      * ```prediction_head_hidden_dim``` (```int```): Number of hidden features of prediction head. Default is ```512```
 
 
* ```train.py```
  * ```get_normalized_std```: Normalizes ```z``` on a per-row basis, gets the standard deviation of the result on a per-row basis, and returns the average of all rows
    * Args:
      * ```z``` (```Tensor```): ```Tensor```
  
  * ```do_training_epoch```: Does one training epoch for a SimSiam model with a provided ```DataLoader``` and returns the loss and the mean normalized standard deviation of projected ```z```s (the latter is useful for monitoring model collapse)
    * Args:
      * ```dataloader``` (```DataLoader```): ```DataLoader``` for training
      * ```model``` (```Module```): SimSiam model
      * ```loss_func``` (```SymmetrizedNegativeCosineSimilarity```): ```SymmetrizedNegativeCosineSimilarity``` object
      * ```optimizer``` (```Optimizer```): ```Optimizer``` for optimization
  
  * ```do_validation```: Validates a SimSiam model with a provided ```DataLoader``` and returns the loss and the mean normalized standard deviation of projected ```z```s (the latter is useful for monitoring model collapse)
    * Args:
      * ```dataloader``` (```DataLoader```): ```DataLoader``` for validation
      * ```model``` (```Module```): SimSiam model
      * ```loss_func``` (```SymmetrizedNegativeCosineSimilarity```): ```SymmetrizedNegativeCosineSimilarity``` object
  
  * ```train```: Trains and validates a SimSiam model and prints loss and standard deviations
    * Args:
      * ```train_dataloader``` (```DataLoader```): ```DataLoader``` for training
      * ```valid_dataloader``` (```DataLoader```): ```DataLoader``` for validation
      * ```model``` (```Module```): SimSiam model
      * ```optimizer``` (```Optimizer```): ```Optimizer``` for optimization
      * ```n_epochs``` (```int```): Number of epochs. Default is ```10```

# Example

The code below trains a ResNet-50 with SimSiam.

```python
from torch.nn import Identity
from torch.optim import Adam
from torchvision.models import resnet50

from data import create_simsiam_dataloader
from model import SimSiamModel
from train import train


# The encoder can be any model that returns a feature vector
encoder = resnet50()
encoder.fc = Identity()

model = SimSiamModel(encoder=encoder,
                     out_dim=2048, 
                     prediction_head_hidden_dim=512)

optimizer = Adam(params=model.parameters(),
                 lr=4e-4)

train_dataloader = create_simsiam_dataloader(path='train/',
                                             valid_exts=['jpeg', 'jpg'],
                                             size=224,
                                             normalize=True,
                                             batch_size=32, 
                                             num_workers=8)
valid_dataloader = create_simsiam_dataloader(path='valid/',
                                             valid_exts=['jpeg', 'jpg'],
                                             size=224,
                                             normalize=True,
                                             batch_size=32, 
                                             num_workers=8)

train(train_dataloader=train_dataloader,
      valid_dataloader=valid_dataloader,
      model=model,
      optimizer=optimizer,
      n_epochs=100)
```
