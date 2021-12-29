import os
import shutil
import random
from classify_func import classify_func
from tqdm import tqdm

test_set_root = 'Data/Data_all'
test_set_tree = os.listdir(test_set_root)

tp, tn, fp, fn = 0, 0, 0, 0

'''
说明
tp----属于数据库中的人，被正确识别为数据库中它的那一类
tn----不属于数据库中的人，被正确识别为 'no matched people'
fp----不属于数据库中的人，被错误识别为数据库中的某一类
fn----属于数据库中的人，被错误识别为不属于它的那一类
'''

all = 0

for single_set in tqdm(test_set_tree):
    single_set_tree = os.listdir(test_set_root + '/' + single_set)
    random.shuffle(single_set_tree)

    for i in range(min(3, len(single_set_tree))):
        classified = classify_func(test_set_root + '/' + single_set + '/' + single_set_tree[i])

        if classified == single_set:
            tp += 1
        else:
            fn += 1
        all += 1

neg_set_root = 'Data/Face Recognition Data/faces95'
neg_set_tree = os.listdir(neg_set_root)

for single_set in tqdm(neg_set_tree):
    single_set_tree = os.listdir(neg_set_root + '/' + single_set)
    random.shuffle(single_set_tree)

    for i in range(min(3, len(single_set_tree))):
        classified = classify_func(neg_set_root + '/' + single_set + '/' + single_set_tree[i])

        if classified == 'no matched people':
            tn += 1
        else:
            fp += 1
        all += 1

precision = float(tp) / (tp + fp)
print("precision = %f"%(precision))
recall = float(tp) / (tp + fn)
print("recall = %f"%(recall))

print('tp = %d'%(tp))
print('tn = %d'%(tn))
print('fp = %d'%(fp))
print('fn = %d'%(fn))