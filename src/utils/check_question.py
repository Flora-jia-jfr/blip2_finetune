import os
import json

eval_file = '/home/shared/MCL/vqav2/v2_mscoco_val2014_annotations.json'
qid = 131089003

with open(eval_file) as f:
    data = f.read()
data=json.loads(data)

qid2instance_dict = {}

for i in data['annotations']:
    qid2instance_dict[i['question_id']] = i

print(qid2instance_dict['131089003'])
