import json
from PIL import Image
import torch
from torchvision import transforms
import argparse, os, sys
from efficientnet_pytorch import EfficientNet
parser=argparse.ArgumentParser()

parser.add_argument('--data_dir', help='Directory with test data.', default = './examples/imagenet/data/')
 
args=parser.parse_args()

model = EfficientNet.from_pretrained('efficientnet-b0')

# Preprocess image
tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

images = [Image.open(args.data_dir + x) for x in os.listdir(args.data_dir)]

for img in images:
    img = tfms(img).unsqueeze(0)

    # Load ImageNet class names
    labels_map = json.load(open('./examples/simple/labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(1000)]

    # Classify
    model.eval()
    with torch.no_grad():
        outputs = model(img)

    # Print predictions
    print('-----')
    for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
        prob = torch.softmax(outputs, dim=1)[0, idx].item()
        print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))




