import torch
import json
import string
import re
import argparse
import sys
from collections import Counter
#import Foundation

from __future__ import print_function

with open('/home/sy/Desktop/KorQuAD_v1.0_train.json') as jsonfile2:
    train=json.load(jsonfile2)  # dict

with open('/home/sy/Desktop/KorQuAD_v1.0_dev.json') as jsonfile:
    validation=json.load(jsonfile) # dict
    
print(f"{len(validation['data'])}") # 140 개 

tmp = []
'''for i in range(140):
    validation['data'][i]'''
print(validation['data'][0]['paragraphs'][0])

def show_ex(data,data_idx):
    ex=data[data_idx]
    #select idxes
    pa_idx=int(input(f'insert paragraph idx(1~{len(ex["paragraphs"])}): '))
    assert (pa_idx>=1)&(pa_idx<=len(ex['paragraphs'])), 'error'
    pa_idx -=1
    ex_pa=ex['paragraphs'][pa_idx]
    qa_idx=int(input(f'insert qa idx (1~{len(ex_pa["qas"])}) : '))
    assert (qa_idx>=1)&(qa_idx<=len(ex_pa['qas'])), 'error'
    qa_idx -=1
    #hightlight
    qas=ex_pa['qas'][qa_idx]
    highlight_idxes=[(x['answer_start'],
                     x['answer_start']+len(x['text']),
                     x['text']) for x in qas['answers']]
    highlight_context=ex_pa['context']
    for (*_,t) in highlight_idxes:
        temp=highlight_context.split(t)
        temp.insert(1, '\033[40;33m]'+t+'\033[m')
        highlight_context=''.join(temp)
        
    print('-'*20)
    print(f'Title:{ex["title"]}')
    print('-'*20)
    print(f'paragraph({pa_idx+1}) context :')
    print(f'{highlight_context}')
    print('-'*20)
    print('1st qa:')
    print(f'question : {qas["question"]}')
    for i,ans in enumerate(qas['answers']):
        print(f'answers:{ans["text"]}')
        print(f'answers start {highlight_idxes[i][0]},end {highlight_idxes[i][1]}')
    return highlight_idxes

with open('/home/sy/Desktop/KorQuAD_v1.0_train.json','r',encoding='utf-8') as file:
    json_string=file.read()
compiler=re.compile('[\W]') # re.compile은 패턴 문자열 pattern을 패턴 객체로 컴파일하는 것  # [\W]는 인덱스 안의 문자열을 의미 
special_token=list(set(compiler.findall(json_string)))  # findall : 정규식과 일치되는 모든 열을 리스트로 리턴
print(special_token)

def find_special(x,spec_token):
    results=[]
    compiler=re.compile('[\W]')
    data_len=len(x)
    for i in range(data_len):
        paras=x[i]['paragraphs']
        paras_len=len(paras)
        for j in range(paras_len):
            context=paras[j]['context']
            special_tokens=compiler.findall(context)
            if spec_token in special_tokens:
                results.append((i,j,context))
    return results    

# korquad1.0에서 제공해주는 some special token in evaluate-v1.0.py
special_token+=[' 《','》 ',' 〈','〉 ','‘','’']
print(special_token)

# 질문과 답변 없이 문제만 있는 context 추출,확인
def check_qas(data,talk=False):
    no_qas=[]
    more_than_two=[]
    for i,article in enumerate(data):
        for j,paras in enumerate(article['paragraphs']):
            context=preprocess_specials(paras['context'])
            qas=paras['qas']
            if len(qas)==0:
                if talk:
                    print(f"{i,j} data no qas")
                no_qas.append((i,j))
            for k,qa in enumerate(qas):
                data_id=qa['id']
                question=qa['question']
                if len(qa['answers'])>=2: 
                    if talk:
                        print(f"{i,j,k} have more than 2 answers")
                    more_than_two.append((i,j,k))
    return no_qas,more_than_two
              
import os
import ujson
from tqdm import tqdm # 진행상태 표시바
              
def preprocess_files(path,file_prename="prepro_"):
    data=ujson.loads(open(path).read())['data']
    temp=[]
    print(len(data))
    for article in tqdm(data,desc="preprocessing",total=len(data)):
        title=article['title']
        
        for paras in article['paragraphs']:
            context=preprocess_specials(paras['context'])
            qas=paras['qas']
            for qa in qas:
                data_id=qa['id']
                question=qa['question']
                for ans in qa['answers']:
                    answer=ans['text']
                    s_idx=ans['answer_start']
                    e_idx=s_idx+len(answer)-1
                    temp.append(dict([('id',data_id),
                                     ('title',title),
                                      ('context',context),
                                      ('question',question),
                                      ('answer',answer),
                                      ('s_idx',s_idx),
                                      ('e_idx',e_idx)
                                     ]))
    print(len(temp))
    save_path=os.path.join(os.path.split(path)[0],file_prename+os.path.split(path)[-1])
    with open(save_path,'w',encoding='utf-8') as file:
        for t in temp:
            ujson.dump(t,file)
            print('',file=file)
        print(f"Done! save to {save_path}")    

path_train='/home/sy/Desktop/KorQuAD_v1.0_train.json'
path_dev='/home/sy/Desktop/KorQuAD_v1.0_dev.json' 
              
preprocess_files(path_train)
preprocess_files(path_dev)              
