import torch
from PIL import Image
from .facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms

def mtcnn_single(image, save_path=None, model=MTCNN(image_size=160, margin=0)):
    #img = image.resize((160, 160))
    img_cropped = model(image, save_path)
    return img_cropped

def embedding_single(model, image):
    img = transforms.ToTensor()(image)
    img_embedding = model(img.unsqueeze(0)).detach().cpu()
    return img_embedding

'''
if __name__ == '__main__':
    mtcnn = MTCNN(image_size=160, margin=0)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    path, save_path = 'D:\Pycharm2020projects\FaceChecking\media\photos\42.png', None
    img = Image.open(path)
    img_cropped = mtcnn_single(img)
    #img_embedding = embedding_single(resnet, img_cropped)
'''