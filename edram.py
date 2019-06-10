import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from torchvision import models
from torchvision.transforms.functional import resize
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
import h5py
from torch.utils.data import Dataset, DataLoader

"""# Loading Data"""
# Please modify this path accordingly
cluster_file = h5py.File('../../../data/mnist-cluttered/mnist_cluttered.hdf5', 'r')

# Setting GPU to 1
print('=================================')
if torch.cuda.is_available():
    device = torch.device("cuda:1")
    torch.cuda.set_device(1)
    print('using GPU: ', device)
else:
    device = torch.device("cpu")
    print('using CPU')
print('=================================')




"""## Loading Data"""

features = np.asarray(np.transpose(cluster_file['features'], [0, 2, 3, 1]), dtype=np.float32)
labels = np.asarray(cluster_file['labels'], dtype=np.int64)
locations = np.asarray(cluster_file['locations'], dtype=np.float32)

train_features = features[0:60000] / 255.
train_labels = labels[0:60000]
train_locations = locations[0:60000]

test_features = features[60000:] / 255.
test_labels = labels[60000:]
test_locations = locations[60000:]

# Check the shape of the dataset
print('shape of train feat',np.shape(train_features))
print('shape of train_labels',np.shape(train_labels))
print('shape of train_locations',np.shape(train_locations))
print('shape of test_features',np.shape(test_features))
print('shape of test_labels',np.shape(test_labels))
print('shape of test_locations',np.shape(test_locations))
print('=================================')

class MyDataset(Dataset):
  def __init__(self, feat, label, location, transform = None):
    self.feat = torch.from_numpy(feat).float()
    self.label = torch.from_numpy(label).int()
    self.location = torch.from_numpy(location).float()
    self.transform = None

  def __getitem__(self, index):
    x = self.feat[index]
    y = self.label[index]
    z = self.location[index]

    if self.transform:
      x = self.transform(x)

    return x, y, z

  def __len__(self):
    return len(self.feat)


batch_size = 128
train_dataset = MyDataset(train_features, train_labels, train_locations)
test_dataset = MyDataset(test_features, test_labels, test_locations)

train_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 4,
    drop_last = True
)

test_loader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    shuffle = False,
    num_workers = 4,
    drop_last = True
)




"""## Helper Function"""

def resize_tensor(input_tensors, h, w):
  final_output = None
  batch_size, channel, height, width = input_tensors.shape
  input_tensors = torch.squeeze(input_tensors, 1)

  for img in input_tensors:
    img = img.cpu().numpy()
    img_PIL = transforms.ToPILImage()(img)
    img_PIL = torchvision.transforms.Resize([h,w])(img_PIL)
    img_PIL = torchvision.transforms.ToTensor()(img_PIL)
    if final_output is None:
      final_output = img_PIL
    else:
      final_output = torch.cat((final_output, img_PIL), 0)
  final_output = torch.unsqueeze(final_output, 1)
  final_output = final_output.to(device)
  return final_output


"""## Building Glimpse Network"""

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x.requires_grad_(True)
        return x.view(x.size(0), -1)


#############################################################################
class GlimpseNet(nn.Module):
  def __init__(self):
    super(GlimpseNet, self).__init__()

    self.layer1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3, padding = 1),
        nn.BatchNorm2d(64),
        nn.ReLU())

    self.layer2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=3),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)),
        nn.ReLU())

    self.layer3 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, padding = 1),
        nn.BatchNorm2d(128),
        nn.ReLU())

    self.layer4 = nn.Sequential(
        nn.Conv2d(128, 128, kernel_size=3, padding = 1),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)),
        nn.ReLU())

    self.layer5 = nn.Sequential(
        nn.Conv2d(128, 160, kernel_size=3, padding = 1),
        nn.BatchNorm2d(160),
        nn.ReLU())

    self.layer6 = nn.Sequential(
        nn.Conv2d(160, 192, kernel_size=3),
        nn.BatchNorm2d(192),
        nn.ReLU())

    self.layer7_image = nn.Sequential(
        Flatten(),
        nn.Linear(3072, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU())

    self.layer8_loc = nn.Sequential(
        nn.Linear(6, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU())
    #initilzing weight and bias on all conv2d layers
    self.layer1[0].weight.data.uniform_(-0.01,0.01)
    self.layer2[0].weight.data.uniform_(-0.01,0.01)
    self.layer3[0].weight.data.uniform_(-0.01,0.01)
    self.layer4[0].weight.data.uniform_(-0.01,0.01)
    self.layer5[0].weight.data.uniform_(-0.01,0.01)
    self.layer6[0].weight.data.uniform_(-0.01,0.01)
    self.layer7_image[1].weight.data.normal_(std = 0.001)
    self.layer8_loc[0].weight.data.normal_(std = 0.001)
    self.layer1[0].bias.data.zero_()
    self.layer2[0].bias.data.zero_()
    self.layer3[0].bias.data.zero_()
    self.layer4[0].bias.data.zero_()
    self.layer5[0].bias.data.zero_()
    self.layer6[0].bias.data.zero_()
    self.layer7_image[1].bias.data.zero_()
    self.layer8_loc[0].bias.data.zero_()



  def forward(self,input_tensors):
    glimpse, theta = input_tensors
    x = self.layer1(glimpse)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.layer6(x)

    # Conv features
    conv_features = self.layer7_image(x)

    # location features

    loc_features = self.layer8_loc(theta)

    return conv_features * loc_features



#############################################################################
## Context Network
class ContextNet(nn.Module):
  def __init__(self, size = 12):
    super(ContextNet, self).__init__()
    self.size = size


    self.layer1 = nn.Sequential(
        nn.Conv2d(1, 16, 5),
        nn.BatchNorm2d(16), ## NoRNN version has a maxpooling here
        nn.ReLU()
    )

    self.layer2 = nn.Sequential(
        nn.Conv2d(16,16,3),
        nn.BatchNorm2d(16),
        nn.ReLU()
    )

    self.layer3 = nn.Sequential(
        nn.Conv2d(16,32,3),
        nn.BatchNorm2d(32), ## NoRNN version has a maxpooling here
        nn.ReLU(),
        Flatten()
    )

    self.layer1[0].weight.data.uniform_(-0.01,0.01)
    self.layer2[0].weight.data.uniform_(-0.01,0.01)
    self.layer3[0].weight.data.uniform_(-0.01,0.01)
    self.layer1[0].bias.data.zero_()
    self.layer2[0].bias.data.zero_()
    self.layer3[0].bias.data.zero_()




  def forward(self, input_tensors):
    resized_image = resize_tensor(input_tensors, 12, 12)
    x = self.layer1(resized_image)
    x = self.layer2(x)
    x = self.layer3(x)
    return x



#############################################################################
# Emission Network
# Remember to add bias later
## Bias and Weight

class EmissionNet(nn.Module):
  def __init__(self):
    super(EmissionNet, self).__init__()
    self.dense1 = nn.Linear(512, 32)
    self.dense2 = nn.Linear(32,6)
    self.Relu = nn.ReLU()
    self.dense2.weight.data.zero_()
    self.dense2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


  def forward(self, input_tensor):
        # Initilize the weights and bias
    x = self.dense1(input_tensor)
    x = self.Relu(x)
    x = self.dense2(x) # there is act functon after the last dense layer
    return x


#############################################################################
class ClassificationNet(nn.Module):
  def __init__(self):
    super(ClassificationNet, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Linear(512,1024),
        nn.BatchNorm1d(1024),
        nn.ReLU()
    )

    self.layer2 = nn.Sequential(
        nn.Linear(1024, 128),
        nn.BatchNorm1d(128),
        nn.ReLU()
    )

    self.layer3 = nn.Sequential(
        nn.Linear(128, 10)
    )

  def forward(self, input_tensor):
    x = self.layer1(input_tensor)
    x = self.layer2(x)
    x = self.layer3(x)
    x = F.log_softmax(x, dim=1)
    return x

#############################################################################
class RecurrentNet(nn.Module):
  def __init__(self):
    super(RecurrentNet, self).__init__()
    self.flatten = Flatten()
    self.class_mlp = nn.Linear(1024, 512 * 4)
    self.class_rnn = nn.LSTMCell(512*4, 512)

    self.emission_mlp = nn.Linear(512, 512 * 4)
    self.emission_rnn = nn.LSTMCell(512 * 4, 512)

    #initilizing weight and bias for RNNs
    self.class_rnn.weight_ih.data.uniform_(-0.01,0.01)
    self.class_rnn.weight_hh.data.uniform_(-0.01,0.01)
    self.class_rnn.bias_ih.data.fill_(0)
    self.class_rnn.bias_hh.data.fill_(0)
    #initilizing weight and bias for dense layers
    self.class_mlp.weight.data.normal_(std = 0.001)
    self.emission_mlp.weight.data.normal_(std = 0.001)
    self.class_mlp.bias.data.zero_()
    self.emission_mlp.bias.data.zero_()

    self.class_state = None
    self.emission_state = None


  def init_hidden(self, batch_size):
    hidden = torch.zeros(batch_size, 512)
    cell = torch.zeros(batch_size, 512)
    hidden = hidden.to(device)
    cell = cell.to(device)
    return (hidden, cell)

  def reset_states(self, conv_init):
    x = torch.zeros_like(conv_init)
    x = x.to(device)
    self.class_state = self.init_hidden(conv_init.shape[0])
    self.emission_state = (conv_init, x)


  def forward(self, input_tensor):
    c = self.class_mlp(input_tensor)
    hidden_class, cell_class = self.class_rnn(c, self.class_state)
    c = hidden_class
    self.class_state = (hidden_class, cell_class)
    e = self.emission_mlp(c)
    hidden_emission, cell_emission = self.emission_rnn(e, self.emission_state)
    e = hidden_emission
    self.emission_state = (hidden_emission, cell_emission)
    return c, e

"""## Bulding EDRAM without RNN"""

class EDRAM(nn.Module):
  def __init__(self, glimpse_size = 26, context_size = 12, batch_size = 128,
              num_glimpse = 6):
    super(EDRAM, self).__init__()
    self.glimpse_size = glimpse_size
    self.context_size = context_size
    self.num_glimpse = num_glimpse

    # Create individual models

    self.glimpse_net = GlimpseNet()
    self.context_net = ContextNet(context_size)
    self.emission_net = EmissionNet()
    self.class_net = ClassificationNet()
    self.rnn_net = RecurrentNet()




  def forward(self, input_tensor):
    probs_array = None
    theta_array = None


    # Note that, the images get resized to 12 * 12 on the contextNet
    shape_input = input_tensor.shape
    input_tensor = input_tensor.view(shape_input[0],shape_input[3],shape_input[1],shape_input[2])

    # context net
    r0 = self.context_net(input_tensor)

    self.rnn_net.reset_states(r0)

    # Get initial glimpse theta
    theta_ori = self.emission_net(r0)
    theta = theta_ori.view(-1, 2, 3)

    # Transform
    grid = F.affine_grid(theta, [batch_size,1,self.glimpse_size,self.glimpse_size])
    glimpse = F.grid_sample(input_tensor, grid)

    for i in range(self.num_glimpse):
      glimpse_features = self.glimpse_net((glimpse, theta_ori))
      r1, r2 = self.rnn_net(glimpse_features)
      probs = self.class_net(r1)

      probs = torch.unsqueeze(probs,0)
      theta_temp = torch.unsqueeze(theta_ori, 0)
      if probs_array is None and theta_array is None:
        probs_array = probs
        theta_array = theta_temp
      else:
        probs_array = torch.cat((probs_array, probs), dim = 0)
        theta_array = torch.cat((theta_array, theta_temp), dim = 0)

      if(i < (self.num_glimpse - 1)):
        theta_ori = self.emission_net(r2)
        theta = theta_ori.view(-1, 2, 3)
        grid = F.affine_grid(theta, [batch_size,1,self.glimpse_size,self.glimpse_size])
        glimpse = F.grid_sample(input_tensor, grid)


    ## saving some for output
    self.glimpse_output = glimpse
    self.input_tensor = input_tensor

    return theta_array, probs_array






###############################################################################

"""## Loss function"""

def accuracy(labels, predicts):
  avg_predicts = torch.mean(predicts,0)
  pred = avg_predicts.max(1, keepdim=True)[1]
  correct = pred.eq(labels.view_as(pred)).sum().item()
  return correct


# def WhereLoss(true_locations, predicted_locations):
#     # Relative importance of where loss to why loss.
#     boost_factor = 1.0
#     loss = (torch.sub(true_locations, predicted_locations)) ** 2
#     scaled_loss = torch.mm(loss, torch.tensor([[1.], [0.5], [1.], [0.5], [1.], [1.]]))

#     return boost_factor * torch.mean(scaled_loss)

def WhereLoss(true_locations, loc_array, glimpse_num = 6):
  # Note: shape of loc_array is 6 (num_glimpse) * 128 * 6 (number of parameter)
    loss_sum = 0
    alpha = torch.tensor([[1.], [0.5], [1.], [0.5], [1.], [1.]])
    alpha = alpha.to(device)

    for glimpse in loc_array: # glimpse is a 128 * 6 tensor
      diff = (torch.sub(glimpse, true_locations)) ** 2
      loss = torch.mm(diff, alpha)
      loss = 1.0 * torch.mean(loss) # average over number of batch (128 here)
      loss_sum += loss
    return loss_sum / glimpse_num # average over number of glimpse


def AverageLoss(labels, probs_array):
  loss_sum = None
  for probs in probs_array:
    loss = F.nll_loss(probs, labels)
    loss = torch.unsqueeze(loss,0)
    if loss_sum is None:
      loss_sum = loss
    else:
      loss_sum = torch.cat((loss_sum, loss), 0)
  return torch.mean(loss_sum)

"""## Training"""

model = EDRAM()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay = 0.05)
# optimizer = optim.SGD(model.parameters(), lr=0.01)

def test():
  with torch.no_grad():
    model.eval()
    test_loss = 0
    correct = 0
    why_loss_all = 0
    where_loss_all = 0
    for feat, lab, loc in test_loader:
          lab = lab.long()
          feat = feat.to(device)
          loc = loc.to(device)
          lab = lab.to(device)
          predicted_locations, preds = model(feat)
          why_loss = AverageLoss(lab, preds)
          where_loss = WhereLoss(loc, predicted_locations)
          loss = why_loss + where_loss
          test_loss += loss
          where_loss_all += where_loss
          why_loss_all += why_loss
          correct += accuracy(lab, preds)
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Why loss: {:.4f}, Where loss: {:.4f}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
      .format(why_loss_all, where_loss_all, test_loss, correct, len(test_loader.dataset),
              100. * correct / len(test_loader.dataset)))



def train(epoch):
  model.train()
  correct = 0
  for batch_idx, (feat, lab, loc) in enumerate(train_loader):
    lab = lab.long()
    feat = feat.to(device)
    lab = lab.to(device)
    loc = loc.to(device)
    optimizer.zero_grad()
    predicted_locations, preds = model(feat)

    why_loss = AverageLoss(lab, preds)
    where_loss = WhereLoss(loc, predicted_locations)
    loss = why_loss + where_loss
    correct += accuracy(lab, preds)
    loss.backward()
    optimizer.step()
    if batch_idx % 10 == 0:
      print("Train Epoch: {} [{} / {} ({:.0f}%)]\tLoss:{:.6f}".format(
              epoch, batch_idx * len(feat),
          len(train_loader.dataset),
          100.* batch_idx / len(train_loader), loss.item()/128))
      print('training accuracy: ', correct / 1280)
      correct = 0




for epoch in range(0, 60):
  train(epoch)
  test()
