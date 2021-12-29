import argparse
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms

def mtcnn_single(args):
    mtcnn = MTCNN(image_size=160, margin=0)

    img = Image.open(args['image_path'])
    img = img.resize((160, 160))
    img_cropped = mtcnn(img, save_path=args['save_path'])

    return img_cropped

def embedding_single(args):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    img = Image.open(args['image_path'])
    img = transforms.ToTensor()(img)
    img_embedding = resnet(img.unsqueeze(0)).detach().cpu()

    return img_embedding

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", help="the path to the image")
ap.add_argument('-s', '--save_path', help='the path to save cropped image', default=None)
args = vars(ap.parse_args())
if __name__ == '__main__':
    mtcnn_single(args)
    print(embedding_single(args))