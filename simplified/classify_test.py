import torch
import json
import argparse
from .facenet_pytorch import InceptionResnetV1
import os
from PIL import Image
from torchvision import transforms


def classify_test(args):
    img = Image.open(args['img_path'])
    img = img.resize((160, 160))
    img = transforms.ToTensor()(img)

    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    img_embedding = resnet(img.unsqueeze(0)).detach().cpu()

    with open(args['dataset_path'], 'r') as f:
        whole_data = json.load(f)
    face_data = whole_data['face_data']
    idx_to_class = whole_data['idx_to_class']

    dists = []
    for i in range(len(face_data)):
        e1 = torch.tensor(face_data[i], dtype=torch.float)
        dist = (e1 - img_embedding).norm().item()
        dists.append(dist)

    #print(dists)
    min_idx = torch.argmin(torch.tensor(dists, dtype=torch.float))
    min_idx = int(min_idx)

    threshold = 0.8
    if dists[min_idx] < threshold:
        name = idx_to_class[str(min_idx)]
        parent_path = args['origin_data'] + '/' + name
        classify_tree = os.listdir(parent_path)
        return parent_path + '/' + classify_tree[0]
    else:
        return 'no matched people'



