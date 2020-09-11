#coding:utf-8
import tensorflow as tf
import logging
import sys
sys.path.append('/home/wujindou/classification_toolkit/')
from data.data_reader_new import DatasetReader
from tensorflow import set_random_seed
set_random_seed(12345)
import sys
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("brc")
    logger.setLevel(logging.INFO)
    brc_data = DatasetReader(train_file='/home/wujindou/dataset/0908/baihuoshipin/train_third.csv',
    #brc_data = DatasetReader(train_file='/home/wujindou/dataset/0905/train_baihuo_category_0905.csv',
                             dev_file='/home/wujindou/dataset/0908/baihuoshipin/dev_third.csv',
                             test_file='/home/wujindou/dataset/0908/baihuoshipin/dev_third.csv',
                             #test_file='/home/wujindou/dataset/test/food_100_category.csv',
                             #test_file='/home/wujindou/dataset/test/food_100_category.csv',
                            # test_file='/home/wujindou/.jupyter/test_single.csv',
                             #test_file='/home/wujindou/dataset/0908/baihuoshipin/dev_third.csv',
                             #test_file='/home/wujindou/dataset/0905/test_baihuo_category_0905.csv',
                             #test_file='/home/wujindou/dataset/0905/test_baihuo_category_0905.csv',
                             #test_file='/home/wujindou/dataset/test_product_category_0827.csv',
                             use_pos_feature=False,
                             prefix='third',
                             use_bert=False
                             )
    from data.vocab import Vocab
    do_inference =True#from data.vocab import Vocab
    vocab = Vocab(lower=True,prefix='third_level_baihuo_')
    if not do_inference:
        for word in brc_data.word_iter(None):
            vocab.add(word)
            for char in word:
                vocab.add_char(char)
        logger.info(' char size {}'.format(vocab.get_char_vocab_size()))
        logger.info(' vocab size {} '.format(vocab.get_word_vocab()))
        #
        unfiltered_vocab_size = vocab.size()
        unfiltered_char_size = vocab.get_char_vocab_size()
        vocab.filter_tokens_by_cnt(min_cnt=2)
        do_inference:vocab.filter_chars_by_cnt(min_cnt=2)
        filtered_num = unfiltered_vocab_size - vocab.size()
        logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                                vocab.size()))

        filtered_num = unfiltered_char_size -vocab.get_char_vocab_size()
        logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                                vocab.get_char_vocab_size()))
        # # sys.exit(1)


    import os
    vocab_file = 'first_third_baihuo_vocab.txt'# vocab.load_from_file('vocab_bool.txt')
    if os.path.exists(vocab_file): vocab.load_from_file(vocab_file)
    if os.path.exists(vocab_file): vocab.load()
    if  not os.path.exists(vocab_file):vocab.load_pretrained_embeddings('/home/wujindou/sgns.merge.word')
    #print(vocab.get_char_vocab_size())#if  not  os.path.exists(vocab_file):vocab.load_pretrained_embeddings('/home/wujindou/sgns.merge.word')
    #sys.exit(1)#print(voab.get_word_vocab())#if  not  os.path.exists(vocab_file):vocab.load_pretrained_embeddings('/home/wujindou/sgns.merge.word')

    if not os.path.exists(vocab_file):
         vocab.save() 
         writer = open(vocab_file, 'a+', encoding='utf-8')
         for word, id in vocab.token2id.items():
             writer.write(word + '\t' + str(id) + '\n')
         writer.close()

    logger.info('after load embedding vocab size is {}'.format(vocab.size()))
    #print(vocab.embeddings.shape)
    import sys
    #sys.exit(1)

    brc_data.convert_to_ids(vocab)

  
    from model.text_cnn import TextCNN
    from model.abilstm import  ABLSTM
    from model.bcnn import  BCNN
    from model.char_cnn import CharCNN
    from model.char_cnn2 import  CharCNN
    from model.bilstm import  BLSTM
    from model.bilstm_gru import  BLSTMGRU
    from model.multi_text_cnn import  MultiTextCNN
    from model.char_word_cnn import  CharTextCNN
    # model = CharTextCNN(vocab,num_class=2,pretrained_word_embedding=vocab.embeddings)
    #model = BLSTMGRU(vocab,num_class=606,pretrained_word_embedding=vocab.embeddings)
    #model = CharTextCNN(vocab,num_class=2)
    #model = ABLSTM(vocab,num_class=606,pretrained_word_embedding=vocab.embeddings)
    #model = ABLSTM(vocab,num_class=606,pretrained_word_embedding=vocab.embeddings)
    #model = TextCNN (vocab,num_class=297,pretrained_word_embedding=vocab.embeddings)
    model = CharTextCNN (vocab,num_class=292,pretrained_word_embedding=vocab.embeddings,word_embedding_size=300)
    #model = TextCNN (vocab,num_class=297,pretrained_word_embedding=vocab.embeddings,word_embedding_size=300)
    #model = TextCNN (vocab,num_class=297,pretrained_word_embedding=vocab.embeddings,word_embedding_size=300)
    #model = TextCNN (vocab,num_class=297,word_embedding_size=300)
    #model = TextCNN (vocab,num_class=69,pretrained_word_embedding=vocab.embeddings)
    #model = TextCNN (vocab,num_class=608,pretrained_word_embedding=vocab.embeddings)
    #model = TextCNN (vocab,num_class=606,pretrained_word_embedding=vocab.embeddings)
    # tf.reset_default_graph()
    # print("----------text cnn -----------")
    #model = CharTextCNN(vocab,pretrained_word_embedding=vocab.embeddings,num_class=2)
    if not do_inference:model.compile(tf.train.AdamOptimizer, 0.001)
     # sys.exit(1)
    if not do_inference:model.train_and_evaluate(brc_data,evaluator=None,epochs=10,save_dir="/home/wujindou/third_level_baihuo_checkpoint",summary_dir='/home/wujindou/summary_writer')
    #
    # tf.reset_default_graph()
    #
    # print("------------ blstm----------- ")
    #
    # model = BLSTM(vocab,pretrained_word_embedding=vocab.embeddings,num_class=3)
    # model.compile(tf.train.AdamOptimizer, 0.001)
    # # sys.exit(1)
    # model.train_and_evaluate(brc_data,evaluator=None,epochs=5,save_dir="checkpoint")
    #
    #
    # tf.reset_default_graph()
    #
    #
    # print("------------ BCNN----------- ")
    #
    # model = BCNN(vocab, pretrained_word_embedding=vocab.embeddings, num_class=3)
    # model.compile(tf.train.AdamOptimizer, 0.001)
    # # sys.exit(1)
    # model.train_and_evaluate(brc_data, evaluator=None, epochs=5,
    #                          save_dir="")
    #
    # tf.reset_default_graph()
    #
    # print("------------ ABLSTM----------- ")
    #
    # model = ABLSTM(vocab, pretrained_word_embedding=vocab.embeddings, num_class=3)
    # model.compile(tf.train.AdamOptimizer, 0.001)
    # # sys.exit(1)
    # model.train_and_evaluate(brc_data, evaluator=None, epochs=5,
    #                          save_dir="")
    #
    # tf.reset_default_graph()

    #
    model.load("/home/wujindou/third_level_baihuo_checkpoint/best_weights")
    model.inference(brc_data,1)

    # from model.bilstm import  BLSTM
    # model = BLSTM(vocab)
    # model.compile(tf.train.AdamOptimizer, 0.001)
    # model.train_and_evaluate(brc_data,evaluator=None,epochs=15)


