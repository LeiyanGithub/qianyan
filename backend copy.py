from fastapi import FastAPI
from anay import similarity
import time

app = FastAPI()

@app.get('/get_similarity/{s1}+{s2}')
def calculate(s1,s2):
    start = time.time()
    answer = similarity(s1,s2)
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
        score.append(similarity(s1,t))
    answer = score.index(max(score))
    res = {"ans": answer}
    end = time.time()
    print("total cost: ",end - start, " s.")
    return res

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app,host='0.0.0.0',port=5000)
