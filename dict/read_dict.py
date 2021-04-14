#coding:utf-8
import sys
import os

file_path = os.path.dirname(__file__)
print(file_path)
def get_dict():
    names = set([line.strip() for line in open(file_path+'/celebrity.csv','r',encoding='utf-8').readlines()])
    return names

movie_name = set([line.strip() for line in open(file_path+'/movies.csv','r',encoding='utf-8').readlines()])

ask_word = set([line.strip().split()[0] for line in open(file_path+'/yiwenci_file','r',encoding='utf-8').readlines()])
sports_names = set([line.strip().split()[0] for line in open(file_path+'/sports','r',encoding='utf-8').readlines()])
names = get_dict()
# for name in sports_names: names.add(name)

# for movie in movie_name:
#     names.add(movie)



if __name__=='__main__':
    i = 0
    writer = open('celebrity','a+',encoding='utf-8')
    with open('./celebrity.csv','r',encoding='utf-8') as lines:
        for line in lines:
            writer.write(line.strip()+'\t'+str(i)+'\n')
            i+=1
    writer.close()
    # print('....')
    # import random
    # data = [line.strip()  for line in open('/Users/apple/Downloads/news_qa/0.7946_test_predict.txt','r',encoding='utf-8').readlines() if int(line.strip().split('\t')[1])==0]
    # samled = random.sample(data,1000)
    # for d in samled:
    #     print(d)
    # text ='薛之谦渣男'
    # import jieba
    # jieba.add_word('薛之谦')
    # print(jieba.lcut(text))
    # tv_actors = set()
    # for name in ['tv_show2.csv']:
    #     with open('/Users/apple/Downloads/news_qa/'+name,'r',encoding='utf-8') as lines:
    #         for line in lines:
    #             if len(line.strip().split(','))<=11:
    #                 print(line)
    #             data = line.strip().split(',')[10]
    #             # print(data)
    #             # sys.exit(1)
    #             tv_actors.add(data)
    #             if 'actor' not in line.strip().split(',')[4]:
    #                 for actor in  line.strip().split(',')[4].split(';'):
    #                     tv_actors.add(actor.strip('"'))
    # for actor in tv_actors:
    #     print(actor.strip('"'))

    # writer = open('names_dict','a+',encoding='utf-8')
    # with open('/Users/apple/Downloads/news_qa/part-00000','r',encoding='utf-8') as lines:
    #     for line in lines:
    #         data = line.strip().split('\t')
    #         if len(data[0])<2: continue
    #         if '剧中人物' in line or '艺人' not in line:
    #             continue
    #         writer.write(data[0]+'\n')#(line.strip())
    # writer.close()