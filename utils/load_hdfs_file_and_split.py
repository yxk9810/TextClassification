#coding:utf-8
import sys
import random
import pydoop.hdfs as hdfs
def read_all_data(file_path='/home/ad/wujindou/text_0908'):
	valid_path = ['part-'+filename.split('part-')[1] for filename in hdfs.ls(file_path) if 'part' in filename ]
	data_all = []
	uniq = set()
	for filename in valid_path:
		with hdfs.open(file_path+'/'+filename) as f:
			for line in f:
				if line.decode() in uniq:continue
				uniq.add(line.decode)
				data_all.append(line.decode())
	import random
	random.shuffle(data_all)
	return data_all
def convert_first_level_data(dataset,level_name='百货食品'):
	data = []
	labels = []
	# level_name = {'美妆饰品':1,'百货食品':2,'服装鞋包':3}
	for d in dataset:
		first_level_name = d.strip().split('\t')[2]
		title = d.strip().split('\t')[-1]
		if first_level_name == level_name:
			label = 1
		else:
			label =0
		labels.append(label)
		data.append(title)
	return (data,labels)
def convert_single_category_level_data(dataset,first_level_name=''):
	data = []
	labels  =[]
	label_map = {}
	for d in dataset:
		level_name=d.strip().split('\t')[2]
		data_arr = d.strip().split('\t')
		if level_name!=first_level_name:continue
		if level_name==u'百货食品' and data_arr[0]=='其他':
			continue
		if data_arr[0] not in label_map:
			label_map[data_arr[0]] = len(label_map)
		data.append(data_arr[-1])
		labels.append(label_map[data_arr[0]])
	from   collections import Counter
	cnt = Counter(labels).most_common(50)
	#print(cnt)
	writer = open('meizhuang_cate_map','a+',encoding="utf-8")
	for cate,id in label_map.items():
		writer.write(cate+'\t'+str(id)+"\n")
	writer.close()
	return (data,labels)

def split_data(x,y,dev_size=0.1):
	from sklearn.model_selection import train_test_split
	trainX, testX, trainY, testY = train_test_split(x, y, test_size=dev_size, random_state=23)
	return (trainX,trainY,testX,testY)
def save_path(d ,folder='/home/wujindou/dataset/0908/baihuoshipin/',level_name='first_level_baihuo'):
	train_x,train_y,test_x,test_y = d
	train_writer = open(folder+'/train_'+level_name+'.csv','a+',encoding="utf-8")
	for x,y in zip(train_x,train_y):
		train_writer.write(x+'\t'+str(y)+"\n")
	train_writer.close()
	dev_writer = open(folder + '/dev_' + level_name + '.csv','a+',encoding="utf-8")
	for x, y in zip(test_x, test_y):
		dev_writer.write(x + '\t' + str(y)+"\n")
	dev_writer.close()

def get_and_split_first_level_data():
	x,y = convert_first_level_data(read_all_data())
	save_path(split_data(x,y))
def get_and_split_third_level_data(first_level_name='美妆饰品'):
	x, y = convert_single_category_level_data(read_all_data(),first_level_name)
	save_path(split_data(x, y),folder='/home/wujindou/dataset/0908/meizhuang/',level_name='third')

if __name__=='__main__':
	get_and_split_first_level_data()
	#get_and_split_third_level_data()
