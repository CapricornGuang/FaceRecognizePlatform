import os
import argparse
import shutil
from tqdm import tqdm
from facenet_pytorch import MTCNN
from PIL import Image

def mtcnn_trans(args):
    data_path = args['image_path']
    data_tree = os.listdir(data_path)

    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=160, margin=0)

    for single_person in tqdm(data_tree):
        data_person_tree = os.listdir(data_path + '/' + single_person)

        for single_face in data_person_tree:
            img_path = data_path + '/' + single_person + '/' + single_face
            img = Image.open(img_path)
            img = img.resize((160, 160))
            img_cropped = mtcnn(img, save_path=img_path)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", help="the path to the image dataset")
args = vars(ap.parse_args())
if __name__ == '__main__':
    mtcnn_trans(args)