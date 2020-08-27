#coding:utf-8
import tensorflow as tf
import logging
import sys
sys.path.append('/home/wujindou/classification_toolkit/')
from data.data_reader_new import DatasetReader
from tensorflow import set_random_seed
set_random_seed(12345)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("brc")
    logger.setLevel(logging.INFO)
    brc_data = DatasetReader(train_file='/home/wujindou/dataset/train_product_category.csv',
                             dev_file = '/home/wujindou/dataset/dev_product_category.csv',
                             bert_dir='/home/wujindou/chinese_L-12_H-768_A-12' , #
                            test_file = None,
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
    #
    unfiltered_vocab_size = vocab.size()
    unfiltered_char_size = vocab.get_char_vocab_size()
    vocab.filter_tokens_by_cnt(min_cnt=2)
    vocab.filter_chars_by_cnt(min_cnt=2)



    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.size()))

    filtered_num = unfiltered_char_size -vocab.get_char_vocab_size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.get_char_vocab_size()))

    logger.info('after load embedding vocab size is {}'.format(vocab.size()))

    brc_data.convert_to_ids(vocab)

    from model.bert_base import BertBaseline

    model = BertBaseline(bert_dir='/home/wujindou/chinese_L-12_H-768_A-12',use_fp16=False,num_class=606)
    num_epoches = 3
    warmup_proportion = 0.1
    num_train_steps = int(
        len(brc_data.train_set) / 32 * num_epoches)
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    model.compile(2e-5, num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps)
    model.train_and_evaluate(brc_data, evaluator=None,
            epochs=num_epoches,save_dir="/home/wujindou/bert_new_checkpoint")



