#coding:utf-8
import sys
import pandas as pd
data_path="/Users/apple/Downloads/embeddings/new_data"
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import  jieba
import random
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
def read_file():
	dataset=[]
	labels = []
	dev_dataset = []
	ids = []
	i = 0
	with open(data_path+"/train_set.csv",'r') as lines:
		for line in lines:
			if i>0:
				data = line.strip().split(",")
				dataset.append(data[2])
				labels.append(int(data[-1])-1)
			i+=1
	j = 0
	with open(data_path+"/test_set.csv",'r') as lines:
		for line in lines:
			if j>0:
				data= line.strip().split(",")
				dev_dataset.append(data[2])
				ids.append(data[0])
			j+=1
	# print(set(labels))
	# sys.exit(1)


	trainX, testX, trainY, testY = train_test_split(dataset, labels, test_size=0.3, random_state=23)
	# writer = open("train_data",'a+')
	# for i in range(len(trainX)):
	# 	writer.write('__label__'+trainY[i]+" "+trainX[i]+"\n")
	# writer.close()
	# writer = open("test_data", 'a+')
	# for i in range(len(testX)):
	# 	writer.write('__label__' + testY[i] + " " + testX[i] + "\n")
	# writer.close()
	# sys.exit(1)



	count_vect = CountVectorizer()
	tfidf_vect = TfidfVectorizer()
	trainX_tfidf = tfidf_vect.fit_transform(trainX)
	testX_tfidf = tfidf_vect.transform(testX)

	trainX = count_vect.fit_transform(trainX)
	testX = count_vect.transform(testX)
	# LR = LogisticRegression()
	# clf = MultinomialNB()
	# #eval_model(LR,trainX,trainY,testX,testY)
	# LR = eval_model(clf,trainX,trainY,testX,testY)
	# eval_model(clf,trainX_tfidf,trainY,testX_tfidf,testY)


	dtrain = xgb.DMatrix(data=trainX,label = trainY)
	ddev = xgb.DMatrix(data=testX,label=testY)
	dtest = xgb.DMatrix(testX)

	param = {"max_depth":6,'eta':0.1}
	param['nthread'] = 8
	param['objective'] = 'multi:softmax'
	param['silent'] =1
	param['num_class'] = 19

	evallist = [(dtrain,'train'),(ddev,'eval')]

	bst = xgb.train(params=param,dtrain=dtrain,num_boost_round=80,evals=evallist,early_stopping_rounds=2)

	y_pred = bst.predict(dtest)

	from sklearn import metrics

	print(classification_report(y_pred,testY))


	# dev_x = count_vect.transform(dev_dataset)
	# pred = LR.predict(dev_x)
	# writer = open("submission.csv",'a+')
	# for i in range(len(pred)):
	# 	writer.write(ids[i]+"\t"+str(pred[i])+"\n")
	# writer.close()

def eval_model(model, trainX, trainY, testX, testY,name='lr'):
    print("---------------result of  "+name+"----------------")
    model.fit(trainX, trainY)
    # print("-----------train data result-----------")
    # train_pred = model.predict(trainX)
    # print(classification_report(train_pred, trainY))
    print("-----------dev data result-------------")
    test_pred = model.predict(testX)

    print(classification_report(test_pred, testY))
    return model
def load_dataset(filename='/Users/apple/Downloads/sentiment_data/train_data_1004.csv',index=0):
	train_x = []
	train_y = []
	with open(filename,"r",encoding="utf-8") as lines:
		for idx,line in enumerate(lines):
			if idx==0:continue
			labels = []  # [int(v) for v in line.strip().split(",")[2:] ]
			for v in line.strip().split(",")[-9:]:
				if v == ' ': v = ''
				if v != '':
					labels.append(int(v) + 1)
				else:
					labels.append(1)
			# print(line.strip().split(","))
			text = ",".join(line.strip().split(",")[:-9])
			if int(labels[index])>= 3: continue
			train_x.append(" ".join(jieba.cut(text, cut_all=False)))
			train_y.append(int(labels[index]))
	return (train_x,train_y)

if __name__=='__main__':

	for i in range(9):
		print('类别 :'+str(i))
		train_x, train_y = load_dataset(index=i)
		test_x, test_y = load_dataset("/Users/apple/Downloads/sentiment_data/test_data_1004.csv",index=i)
		count_vect = CountVectorizer()
		train_x = count_vect.fit_transform(train_x)
		test_x = count_vect.transform(test_x)
		LR = LogisticRegression()
		eval_model(LR,train_x,train_y,test_x,test_y)
		clf = MultinomialNB()
		eval_model(LR,train_x,train_y,test_x,test_y,name="multinomial naive bayies")
		from sklearn.svm import SVC

		clf = SVC(kernel='linear')

		eval_model(clf,train_x,train_y,test_x,test_y,name="svm")



