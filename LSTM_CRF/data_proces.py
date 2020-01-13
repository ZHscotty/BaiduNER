import os
import json

"""
观察哪些样本中存在，某个实体既做主体又做客体
"""
f = open('./原始数据/train_data.json', 'r', encoding='utf-8')
fj = open('./data/交集.txt', 'w', encoding='utf-8')
index = 0
for line in f.readlines():
    print('Processing text:', index)
    j = json.loads(line)
    Object = [o['object'] for o in j['spo_list']]
    Subject = [s['subject'] for s in j['spo_list']]
    intersection = set(Object) & set(Subject)
    if len(intersection) != 0:
        fj.write('文本'+str(index)+'\t'+'\t'.join(intersection)+'\n')
    index += 1

f.close()
fj.close()