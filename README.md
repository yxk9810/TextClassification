# TextClassification

#模型包括：

 text_cnn/char-cnn/bilstm/ablstm/Bert/bcnn和借鉴的一些kaggle上的模型


#效果，一些实际测试显示text cnn一般效果不错



#trick
####bert模型一些实际中使用效果是不错，但是由于速度问题，可能并不适合上线任务，可以通过模型蒸馏利用无标签数据进行。
1.在有标签数据上训练Bert
2.bert预测无标签数据,得到soft lable 和hard label
3.text cnn在无标签数据上进行训练，以bert预测的hard label或者soft label为true label
4.tex cnn在有标签数据上进行训练。
测试效果：
| text cnn   | bert      |  蒸馏     |
| --------   | -----:    | :----:   |
| 0.86/0.82  | 0.86/0.86 | 0.87/0.84|




