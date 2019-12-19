#coding:utf-8
import tensorflow as tf
import logging
from data.data_reader_distill import DatasetReaderDistill
from tensorflow import set_random_seed
set_random_seed(12345)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("brc")
    logger.setLevel(logging.INFO)
    brc_data = DatasetReaderDistill(train_file='./dataset/seg_train_data_20w.txt',
                             dev_file='./dataset/seg_dev_data_20w.txt',
                             # test_file ='./dataset/test_data'
                             )
    from data.vocab import Vocab

    vocab = Vocab(lower=True)
    import sys
    for word in brc_data.word_iter(None):
        vocab.add(word)
        for char in word:
            vocab.add_char(char)
    logger.info(' char size {}'.format(vocab.get_char_vocab_size()))
    logger.info(' vocab size {} '.format(vocab.get_word_vocab()))

    unfiltered_vocab_size = vocab.size()
    unfiltered_char_size = vocab.get_char_vocab_size()
    vocab.filter_tokens_by_cnt(min_cnt=2)
    vocab.filter_chars_by_cnt(min_cnt=2)

    import os
    if not os.path.exists('vocab.txt'):
        writer = open('vocab.txt','a+',encoding='utf-8')
        for word,id in vocab.token2id.items():
            writer.write(word+'\t'+str(id)+'\n')
        writer.close()

    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.size()))

    filtered_num = unfiltered_char_size -vocab.get_char_vocab_size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.get_char_vocab_size()))
    # sys.exit(1)

    #vocab.load_pretrained_embeddings('/Users/apple/Downloads/embeddings/sgns.wiki.word')

    logger.info('after load embedding vocab size is {}'.format(vocab.size()))
    # print(vocab.embeddings.shape)
    # import sys
    # sys.exit(1)

    brc_data.convert_to_ids(vocab)
    import numpy as np
    for task_balance in np.arange(0.12,0.51,0.01):

        from model.text_cnn import TextCNN
        from model.abilstm import  ABLSTM
        from model.bcnn import  BCNN
        # from model.char_cnn import CharCNN
        from model.char_cnn2 import  CharCNN
        from model.bilstm import  BLSTM
        from model.multi_text_cnn import  MultiTextCNN
        from model.char_word_cnn import  CharTextCNN
        #model = CharCNN(vocab,num_class=2)
        #model = BCNN(vocab,num_class=2)
        #model = CharTextCNN(vocab,num_class=2)
        #model = ABLSTM(vocab,num_class=2)
        #model = BLSTM(vocab,num_class=2)
        tf.reset_default_graph()
        save_dir = '/Users/apple/Downloads/news_qa/checkpoint'
        model = TextCNN(vocab,num_class=3,task_balance=0.12,soft_temperature=10)
        model.compile(tf.train.AdamOptimizer, 0.001)
        model.load('/Users/apple/Downloads/news_qa/pretrained_checkpoint/best_weights/')
        model.train_and_evaluate(brc_data, evaluator=None, epochs=5, save_dir=save_dir)
        sys.exit(1)

        if task_balance==1:
            model.train_and_evaluate(brc_data,evaluator=None,epochs=5,save_dir=save_dir)
        else:
            model.load('/Users/apple/Downloads/news_qa/checkpoint/best_weights/')
            model.train_and_evaluate(brc_data,evaluator=None,epochs=5,save_dir=save_dir)

        print('..........finish training with  {} ............'.format(task_balance))

    # from model.bilstm import  BLSTM
    # model = BLSTM(vocab)
    # model.compile(tf.train.AdamOptimizer, 0.001)
    # model.train_and_evaluate(brc_data,evaluator=None,epochs=15)


