import torch
import json
import argparse
from .facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms

def face_append(args):
    img = Image.open(args['image_path'])
    img = img.resize((160, 160))
    img = transforms.ToTensor()(img)

    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    img_embedding = resnet(img.unsqueeze(0)).detach().cpu()

    with open(args['dataset_path'], 'r') as f:
        whole_data = json.load(f)
    face_data = whole_data['face_data']
    idx_to_class = whole_data['idx_to_class']

    idx_to_class[str(len(idx_to_class))] = args['name']
    face_data.append(img_embedding.tolist())

    whole_data = {'face_data': face_data, 'idx_to_class': idx_to_class}
    with open(args['dataset_path'], 'w') as f:
        json.dump(whole_data, f)


if __name__ == '__main__':
    face_append(args)