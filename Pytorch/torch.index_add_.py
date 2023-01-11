import numpy as np
import random
import torch
if __name__ == '__main__':
    '''
    The problem is still open:
    Sorted the index but still non-deterministic
    '''

    n=50
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda")
    # Create some data
    src = torch.randn((n,n)).to(device)
    dst = torch.zeros((n,n)).to(device)
    indices = torch.randint(0, n, (n,), dtype=torch.int).to(device)
    print("indices",indices)
    indices,_=torch.sort(indices)
    print("sorted indices",indices)
    # Use the indices to update the destination tensor
    dst1=dst.index_add_(0, indices, src)

    # Do the same thing again
    dst = torch.zeros((n,n)).to(device)
    dst2=dst.index_add_(0, indices,src)

    print("-"*10)
    torch.set_printoptions(precision=8)
    print("differences between run 1 and run 2")
    for x,y in zip(dst1,dst2):
        if not torch.all(x==y):
            print(x)
            print(y)
            print(x==y)

    #print overall comparison
    print("-" * 10)
    compare=dst1.flatten()==dst2.flatten()
    print(torch.all(compare))
