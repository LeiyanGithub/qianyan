from fastapi import FastAPI
from anay import similarity, convert_examples_to_features, InputExample, NeuralNet, select_field
import time
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import AdamW
from transformers import BertTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.cuda.amp import autocast, GradScaler
from torch.optim.optimizer import Optimizer
import os

# 设置参数及文件路径
os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'
max_seq_length = 128  # 输入文本最大长度
batch_size = 1  # 训练时每个batch中的样本数
file_name = 'baseline'  # 指定输出文件的名字
model_name_or_path = './pretrain_models/ernie_gram'  # 预训练模型权重载入路径
app = FastAPI()
tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)



model = NeuralNet(model_name_or_path)
# model = nn.DataParallel(model, device_ids=[0, 1 , 2 , 3])
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.cuda()

# 得到一折模型对测试集的预测结果
model.load_state_dict(torch.load('./model_save/ernie_' + file_name + '_{}.bin'.format(0)),False)
model.eval()

@app.get('/get_similarity/{s1}+{s2}')
def calculate(s1,s2):
    start = time.time()
    
    answer = similarity(s1, s2, model, tokenizer)
    res = {"ans": answer}
    end = time.time()
    print("total cost: ",end - start, " s.")
    return res

@app.get('/get_topic/{s1}')
def calcu(s1):
    start = time.time()
    topics = ['更改手机号','官方客服电话','开户失败','贷款失败','审批条件']
    score = []
    for t in topics:
        score.append(similarity(s1,t, model, tokenizer))
    answer = score.index(max(score))
    res = {"ans": answer}
    end = time.time()
    print("total cost: ",end - start, " s.")
    return res

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app,host='0.0.0.0',port=5000)
