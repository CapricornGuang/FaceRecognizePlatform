import os
import shutil
from tqdm import tqdm

Recognized_Data_tree = os.listdir('Data/Data_all')
Recognized_Data = 'Data/Data_all'
In_dataset = 'Data/Dataset_all'

for single_person in tqdm(Recognized_Data_tree):
    first_file_name = os.listdir(Recognized_Data + '/' + single_person)[0]
    first_file_path = Recognized_Data + '/' + single_person + '/' + first_file_name

    os.mkdir(In_dataset + '/' + single_person)
    des_file_path = In_dataset + '/' + single_person + '/' + first_file_name

    shutil.move(first_file_path, des_file_path)