'''
工作流程说明：
1、批量导入人脸信息。每个人一张照片，且放在对应人名的文件夹中
2、批量进行人脸编码。即将人脸输入模型，得到对应编码向量
3、将各个人脸编码存入文件中，待检测人脸时使用
'''

import argparse
import json
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm


def face_in(args):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    transform = transforms.Compose([
        transforms.Resize(160),
        transforms.ToTensor() #将图片转换为Tensor,归一化至[0,1]
    ])

    dataset = datasets.ImageFolder(args['image_path'], transform=transform)
    class_to_idx = dataset.class_to_idx
    idx_to_class = {}

    for key,value in class_to_idx.items():
        idx_to_class[value] = key

    total_num = len(dataset)

    face_data = []
    for i in tqdm(range(total_num)):
        img, idx = dataset[i]
        img_embedding = resnet(img.unsqueeze(0)).detach().cpu()
        #print(img_embedding.squeeze(0).shape)

        face_data.append(img_embedding.tolist())

    #e1 = torch.tensor(face_data[0], dtype=torch.long)
    #print(e1.shape)
    whole_data = {'face_data': face_data, 'idx_to_class': idx_to_class}
    with open(args['dataset_path'], 'w') as f:
        json.dump(whole_data, f)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", help="the path to the image dataset")
ap.add_argument('-d', '--dataset_path', help='the name of dataset, with suffix please',
                default='dataset.json')
args = vars(ap.parse_args())
if __name__ == '__main__':
    face_in(args)