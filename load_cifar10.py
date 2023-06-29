from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import glob
from PIL import Image

label_name = ["airplane",
              "automobile",
              "bird",
              "cat",
              "deer",
              "dog",
              "frog",
              "horse",
              "ship",
              "truck"]

label_dict = {}
for idx, label in enumerate(label_name):
    label_dict[label] = idx


def default_loader(path):
    return Image.open(path).convert('RGB')

# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop((28, 28)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(90),
#     transforms.RandomGrayscale(0.1),
#     transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
#     transforms.ToTensor()
# ])


train_transform = transforms.Compose([
    transforms.RandomCrop(28),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


class MyDataset(Dataset):
    def __init__(self, im_list, transform=None, loader=default_loader):
        super(MyDataset, self).__init__()

        imgs = []

        # '/home/hanhan3344/Study/cv/TRAIN/airplane/aeroplane_s_000004.png'
        for im_item in im_list:
            im_label_name = im_item.split("/")[-2]
            imgs.append((im_item, label_dict[im_label_name]))

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):  # 定义对数据的读取、数据的增强,返回图片的数据和label在训练中会反复调用
        im_path, im_label = self.imgs[index]

        im_data = self.loader(im_path)

        if self.transform is not None:  # transform 在训练和测试的时候是不一样的
            im_data = self.transform(im_data)

        return im_data, im_label

    def __len__(self):
        return len(self.imgs)


im_train_list = glob.glob('TRAIN/*/*.png')
im_test_list = glob.glob('TEST/*/*.png')

train_dataset = MyDataset(im_train_list, transform=train_transform)
test_dataset = MyDataset(im_test_list, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset,
                          batch_size=128,
                          shuffle=True,
                          num_workers=4)

test_loader = DataLoader(test_dataset,
                         batch_size=128,
                         shuffle=False,
                         num_workers=4)

print("num of train", len(train_dataset))
print("num of test", len(test_dataset))
