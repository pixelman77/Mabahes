import torch 
from torchvision import transforms
from torchvision import models
from os import listdir
from os.path import isfile, join
from PIL import Image

expected = 719 #expected value

correct = 0
amount = 0

path = "pictures/"

with open("classes.txt", "r") as f:
  categories = [s.strip() for s in f.readlines()]

transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225])])

model = models.alexnet(pretrained=True)

onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

for fl in onlyfiles:
  amount += 1


  image = Image.open(path + fl)
  img_tensor = transform(image)
  input_batch = img_tensor.unsqueeze(0)
  
  if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

  with torch.no_grad():
    output = model(input_batch)

  probabilities = torch.nn.functional.softmax(output[0], dim=0)


  top5_prob, top5_catid = torch.topk(probabilities, 5)
  
  for i in range(top5_prob.size(0)):

    if(expected.__str__() in categories[top5_catid[i]]):
      correct += 1

    print(categories[top5_catid[i]], top5_prob[i].item())
  
print("\ntotal amount: " + amount.__str__() + "\ncorrect results: " + correct.__str__() + "\naccuracy: " + (correct/amount * 100).__str__())