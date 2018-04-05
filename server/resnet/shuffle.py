import os
import random


for i in range(88,89):
    source_path = './chunks_1000s/train______________' + str(i) + '.txt'
    target_path = './chunks_1000s/train_______________' + str(i) + '.txt'
    with open(source_path,'r') as source:
        data = [(random.random(), line) for line in source]
    data.sort()
    with open(target_path,'w') as target:
        for _, line in data:
            target.write(line)
    print(str(i))