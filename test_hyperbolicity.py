from torch.utils.data import TensorDataset, DataLoader
from hyptorch.datasetSALICON import SALICON
import torchvision
import torchvision.transforms as transforms
import hyptorch.delta as delta
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from PIL import Image

def test_salicon_hyperbolicity():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = DataLoader(
        SALICON(mode='train', N=100),
        batch_size=10,
        shuffle=True,
        **kwargs
    )

    return delta.get_delta(train_loader)

def test_cifar_hyperbolicity():
    use_cuda = torch.cuda.is_available()
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    transform = transforms.Compose(
        [transforms.Resize((224,224),interpolation=Image.NEAREST), transforms.ToTensor()])

    batch_size = 32
    
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    odds = list(range(1, len(trainset), 2))
    trainset_1 = torch.utils.data.Subset(trainset, evens)
    trainloader = torch.utils.data.DataLoader(trainset_1, batch_size=batch_size,
                                              shuffle=True, num_workers=2)


    # for images, _ in trainloader:
    #     print('images.shape:', images.shape)
    #     plt.figure(figsize=(16, 8))
    #     plt.axis('off')
    #     plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    #     break

    return delta.get_delta(trainloader)


def run():
    torch.multiprocessing.freeze_support()
    print( "the hyperbolicity of cifar is: " + str(test_cifar_hyperbolicity()))
    print("the hyperbolicity of SALICON is: " + str(test_salicon_hyperbolicity()))
    print( "the hyperbolicity of cifar is: " + str(test_cifar_hyperbolicity()))

if __name__ == '__main__':
    run()