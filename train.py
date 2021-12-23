from typing import Tuple

from torch import no_grad, std, Tensor
from torch.nn import Module
from torch.nn.functional import normalize
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from loss import SymmetrizedNegativeCosineSimilarity


def get_normalized_std(z: Tensor) -> float:
    """
    Returns the average of the standard deviation of normalized z
    
    Args:
        z (Tensor): Tensor
    """
    with no_grad():
        z_normalized = normalize(z, dim=1)
        z_std = std(z_normalized, dim=1)
        mean_std = z_std.mean()
    return mean_std


def do_training_epoch(dataloader: DataLoader, model: Module, 
                      loss_func: SymmetrizedNegativeCosineSimilarity,
                      optimizer: Optimizer) -> Tuple[float, float, float]:
    """
    Does one training epoch for SimSiam model and returns loss plus the mean
    normalized standard deviation of projected tensors

    Args:
        dataloader (DataLoader): SimSiam DataLoader for training
        model (Module): SimSiam model
        loss_func (SymmetrizedNegativeCosineSimilarity): SymmetrizedNegativeCosineSimilarity object
        optimizer (Optimizer): Optimizer
    """
    model = model.train()

    total_loss = 0.0
    total_std1 = 0.0
    total_std2 = 0.0

    for x1, x2 in dataloader:
        optimizer.zero_grad()

        x1 = x1.cuda()
        x2 = x2.cuda()

        z1, p1 = model(x1)
        z2, p2 = model(x2)

        loss = loss_func(p1, z1, p2, z2)
        loss.backward()

        optimizer.step()

        total_loss += len(x1)*loss.item()
        total_std1 += len(x1)*get_normalized_std(z1)
        total_std1 += len(x1)*get_normalized_std(z2)
    
    n_data_points = len(dataloader.dataset)
    total_loss /= n_data_points
    total_std1 /= n_data_points
    total_std2 /= n_data_points
    return total_loss, total_std1, total_std2


def do_validation(dataloader: DataLoader, model: Module,
                  loss_func: SymmetrizedNegativeCosineSimilarity
                  ) -> Tuple[float, float, float]:
    """
    Validates SimSiam model and returns loss plus the mean normalized standard
    deviation of projected tensors

    Args:
        dataloader (DataLoader): SimSiam DataLoader for validation
        model (Module): SimSiam model
        loss_func (SymmetrizedNegativeCosineSimilarity): SymmetrizedNegativeCosineSimilarity object
    """
    model = model.eval()

    total_loss = 0.0
    total_std1 = 0.0
    total_std2 = 0.0
    
    with no_grad():
        for x1, x2 in dataloader:
            x1 = x1.cuda()
            x2 = x2.cuda()

            z1, p1 = model(x1)
            z2, p2 = model(x2)

            loss = loss_func(p1, z1, p2, z2)

            total_loss += len(x1)*loss.item()
            total_std1 += len(x1)*get_normalized_std(z1)
            total_std1 += len(x1)*get_normalized_std(z2)
    
    n_data_points = len(dataloader.dataset)
    total_loss /= n_data_points
    total_std1 /= n_data_points
    total_std2 /= n_data_points
    return total_loss, total_std1, total_std2


def train(train_dataloader: DataLoader, valid_dataloader: DataLoader, 
          model: Module, optimizer: Optimizer, n_epochs: int = 10):
    """
    Trains and validates SimSiam model

    Args:
        train_dataloader (DataLoader): SimSiam DataLoader for training
        valid_dataloader (DataLoader): SimSiam DataLoader for validation
        model (Module): SimSiam model
        optimizer (Optimizer): Optimizer
        n_epochs (int): Number of epochs
    """
    model = model.cuda()
    loss_func = SymmetrizedNegativeCosineSimilarity()

    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')

        total_loss, total_std1, total_std2 = do_training_epoch(dataloader=train_dataloader, 
                                                               model=model, 
                                                               loss_func=loss_func,
                                                               optimizer=optimizer)
        
        print(f'Training loss: {total_loss}')
        print(f'Training STD1: {total_std1}')
        print(f'Training STD2: {total_std2}')


        total_loss, total_std1, total_std2 = do_validation(dataloader=valid_dataloader, 
                                                           model=model, 
                                                           loss_func=loss_func)

        print(f'Validation loss: {total_loss}')
        print(f'Validation STD1: {total_std1}')
        print(f'Validation STD2: {total_std2}')
