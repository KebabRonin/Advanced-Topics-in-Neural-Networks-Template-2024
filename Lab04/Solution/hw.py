import torch, parse, os, sys, torchvision, random
import torchvision.transforms as transf
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision.transforms import ToTensor


class MyDataset(Dataset):
  FILE_NAME_FORMAT = 'global_monthly_{year:d}_{month:d}_mosaic_{loc}.tif'
  def __init__(self, path, transforms=None):
    self.path = path
    if transforms == None:
      self.transforms = lambda x: x
    else:
      self.transforms = transforms
    self.locations = sorted(os.listdir(self.path))
    self.data = [
      (l, sorted([
        (
          parse.parse(MyDataset.FILE_NAME_FORMAT, t)['year'] * 12 +
              parse.parse(MyDataset.FILE_NAME_FORMAT, t)['month'],
          ToTensor()(Image.open(f"{self.path}/{l}/images/{t}")) # Images are cached here
        )
        for t in os.listdir(f"{self.path}/{l}/images")
      ], key=lambda x: x[0]))
      for l in self.locations
    ]

    # Precompute indices of locations
    self.cumulative_idx = [0]
    s = 0
    for k in self.data:
      s += (len(k[1]) * (len(k[1]) - 1)) // 2
      self.cumulative_idx.append(s)


  def __len__(self):
    return self.cumulative_idx[-1]


  def find_loc(self, idx):
    """
    Get the location associated with the idx th pair in the dataset
    """
    ii = 0
    for i, val in enumerate(self.cumulative_idx):
      if val < idx:
        ii = i
      else:
        return ii
    return ii

  def resolve_index(self, idx):
    """
    Returns the pair of images described by the index
    """
    loc_idx = self.find_loc(idx)
    loc = self.data[loc_idx][1]
    # print(self.data[loc_idx][0])
    loc_len = len(loc)
    # Get permutation index
    idx -= self.cumulative_idx[loc_idx] # Search in possible permutations of the current location
    idx2 = idx
    idx1 = 0
    while True:
      c_len = loc_len - idx1 - 1
      if idx2 < c_len:
        idx2 += idx1 + 1 # idx2 is the offset from idx1 in the future
        break
      else:
        idx2 -= c_len
        idx1 += 1
    return loc[idx1], loc[idx2]

  def __getitem__(self, idx):
    im1, im2 = self.resolve_index(idx)
    rstate = random.getstate()
    state = torch.get_rng_state() # Apply same transform
    transformed1 = self.transforms(im1[1])
    torch.set_rng_state(state)
    random.setstate(rstate)
    transformed2 = self.transforms(im2[1])
    return (
      transformed1,
      transformed2,
      im2[0] - im1[0] # Difference in months
    )

dset_path = '../Dataset'

ds = MyDataset(
  f"{dset_path}_train",
  # transforms=transf.RandomRotation(180)
)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# This splits uniformly, but could also lead to overfitting (the model sees all locations)
# ds_train, ds_valid, ds_test = random_split(ds, [0.7, 0.15, 0.15])

# This might work better: 42 / 9 / 9 folders from the original 60.
# Data is missing from some, so maybe not truly 70%/15%/15%, but it is what it is.
ds_train, ds_valid, ds_test = MyDataset(f"{dset_path}_train"), MyDataset(f"{dset_path}_valid"), MyDataset(f"{dset_path}_test")

train_dloader = DataLoader(ds_train, batch_size=64, shuffle=True)
valid_dloader = DataLoader(ds_valid, batch_size=256, shuffle=False)
test_dloader  = DataLoader(ds_test, batch_size=256, shuffle=False)


import matplotlib.pyplot as plt

def view_dset(ds):
  for im1, im2, _ in random.choices(ds, k=30):
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(im1.permute(1, 2, 0))
    axarr[1].imshow(im2.permute(1, 2, 0))

    plt.show()


view_dset(ds_train)