# coding:utf-8

# to do jieba
import jieba
from collections import Counter
import logging
import numpy as np
np.random.seed(12345)
import sys
import re
from utils.proprocess import clean_text, clean_numbers,clean_date
# from utils.clean_tool import clean_title
# from dict.read_dict import *
# for name in names: jieba.add_word(name)
import pickle
names = set()
import os
ask_word = set()
from libraries.BertWrapper import BertDataHelper

class DatasetReader(object):
    def __init__(self, data_file=None, train_file=None, dev_file=None,test_file =None,use_name_feature=True,use_pos_feature=False,use_char=True,use_bert=True,
                 bert_dir = '/Users/apple/Downloads/chinese_L-12_H-768_A-12',label_index = 0,max_seq_len=20,prefix=''):
        self.max_len = 20
        self.max_c_len = 40
        self.max_seq_len =max_seq_len
        self.dataset = []
        self.test_set = []
        self.prefix =prefix 
        self.use_name_feature = use_name_feature
        self.use_pos_feature = use_pos_feature
        self.use_char = use_char
        self.use_splited_token=False
        self.train_sentiment = False
        self.use_bert = use_bert
        self.bert_helper = None
        if self.use_bert:
            self.bert_helper =  BertDataHelper(bert_dir)
        if self.train_sentiment :
            self.label_index = label_index


        self.logger = logging.getLogger("brc")
        if data_file:
            self.dataset += self._load_dataset(data_file)
            self.logger.info('data set size: {} '.format(len(self.dataset)))
            import numpy as np
            np.random.seed(10)
            shuffle_indices = np.random.permutation(np.arange(len(self.dataset)))

            self.dataset = np.array(self.dataset)[shuffle_indices].tolist()
            dataset_size = len(self.dataset)

            # Randomly shuffle data
            # random.shuffle(self.dataset)
            dataset_shuffled = self.dataset

            # Split train/test set
            # TODO: This is very crude, should use cross-validation
            dev_sample_index = -1 * int(.1 * float(len(dataset_shuffled)))

            self.train_set = dataset_shuffled[:dev_sample_index]
            train_writer = open('../dataset/train_news_0827_data.txt','a+',encoding='utf-8')
            for sample in self.train_set:
                train_writer.write(sample['raw_text']+'\t'+str(sample['label'])+'\n')
            train_writer.close()

            self.dev_set = dataset_shuffled[dev_sample_index:]
            dev_writer = open('../dataset/dev_news_0827_data.txt', 'a+', encoding='utf-8')
            for sample in self.dev_set:
                dev_writer.write(sample['raw_text'] + '\t' + str(sample['label']) + '\n')
            dev_writer.close()
            sys.exit(1)

        if train_file:
            train_save_pk_file = self.prefix+'train.pkl'
            if not os.path.exists(train_save_pk_file):
                self.train_set = self._load_dataset(train_file,is_train=True)
                with open(train_save_pk_file,'wb') as f:
                    pickle.dump(self.train_set,f)
            else:
                with open(train_save_pk_file,'rb') as f:
                    self.train_set =pickle.load(f)
                    print("load train set size="+str(len(self.train_set)))
        if dev_file:
            dev_save_pk_file = self.prefix+'dev.pkl'
            if not os.path.exists(dev_save_pk_file):
                self.dev_set = self._load_dataset(dev_file)

                with open(dev_save_pk_file, 'wb') as f:
                    pickle.dump(self.dev_set, f)
            else:
                with open(dev_save_pk_file,'rb') as f:
                    self.dev_set = pickle.load(f)
                    print("loading dev set size="+str(len(self.train_set)))

        if test_file:
            self.test_set = self._load_dataset(test_file,is_test=True)

        #print(self.train_set)#:self.logger.info('train set size: {} '.format(len(self.train_set)))
        if train_file:self.logger.info('train set size: {} '.format(len(self.train_set)))
        if dev_file:self.logger.info('dev set size: {}'.format(len(self.dev_set)))
        if test_file:self.logger.info('test set size: {}'.format(len(self.test_set)))

        if not train_file and not dev_file: self.train_set, self.dev_set, = [], []
        # if train_files:
        #     for train_file in train_files:
        #         self.train_set += self._load_dataset(train_file, train=True)
        #     self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))
        #
        # if dev_files:
        #     for dev_file in dev_files:
        #         self.dev_set += self._load_dataset(dev_file)
        #     self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))
        #
        # if test_files:
        #     for test_file in test_files:
        #         self.test_set += self._load_dataset(test_file)
        #     self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))
        #

    '''
    by default we assume the format of file : 
    context \t label 
    '''

    def _load_dataset(self, filename, use_char=False,is_test=False,is_train=False):
        dataset = []
        y = []
        pre = ''
        index = 0 
        with open(filename, 'r', encoding='utf-8') as lines:
            for index,line in enumerate(lines):
                if index>=2000 and is_train :break
                # if line.strip() == '' or len(line.strip()) == 0:
                #     continue
                item_id = None
                if self.use_pos_feature:
                    data = line.strip().split('\t')
                    tokens = [w.split('#pos#')[0] for w in data[0].split()]
                    text =''.join(tokens)
                    tokens = [token if not token.isnumeric() and token.strip() != '' and not token.replace('.', '', 1).isdigit() else 'digit' for token in tokens]
                    tokens = [token if re.match('^(\d{1,2}月$|\d{1,2}(日|号)$|\d{4}年$)',token) is None else 'date' for token in  tokens]
                    char_tokens = []
                    for token in tokens:
                        char_tokens.extend(list(token))
                    if use_char:
                        tokens = char_tokens
                    token_pos = [int(w.split('#pos#')[1]) for w in data[0].split()]
                    if len(data)==1: label = 0
                    else:
                        label =int(data[1])
                elif self.use_splited_token:
                    data = line.strip().split('\t')
                    tokens = data[1].strip().split(' ')
                    char_tokens = []
                    for token in tokens:
                        char_tokens.extend(list(token))
                    label = int(data[-2].split(' fake: ')[0].strip())
                    text=''.join(data[1].strip().split(' '))
                elif self.train_sentiment:
                    labels =[] #[int(v) for v in line.strip().split(",")[2:] ]
                    for v in line.strip().split(",")[-9:]:
                        if v == ' ':v = ''
                        if v!='':labels.append(int(v)+1)
                        else:
                            labels.append(1)
                    # print(line.strip().split(","))
                    text =",".join(line.strip().split(",")[:-9])
                    label = labels[self.label_index]
                    if int(label)>=3:continue
                    tokens = [token for token in jieba.lcut(clean_date(text)) if token.strip()!='']
                    char_tokens = []
                    for token in jieba.lcut(clean_numbers(text)):
                        char_tokens.extend(list(token))


                else:
                    data = line.strip().split('\t')
                    if not is_test and len(data)<2:continue# = line.strip().split('\t')
                    text  = data[0]
                    label = int(data[1]) if not is_test else 0
                    import re
                    clean_title_str = re.sub(r'闪电购商品[\s\d+]{0,}', '', text.strip())
                    if len(clean_title_str) < 2: continue
                    # if len(clean_title(clean_title_str).strip())==0:continue#if len(clean_title_str) < 2: continue
                    if is_test:
                        if len(data)<2:continue
                        item_id = data[1]

                    #valid_labels=[0,2,3,5,12,20,28,45,47,48,50,56,58,61,64,66,77,82,87,93,95,98,100,105,106,111,116,117,123,125,128,131,142,144,148,154,158,160,162,164,168,171,172,177,180,193,202,203,209,211,215,222,223,229,241,243,245,248,249,250,252,260,265,268,271,278,281,283,287,289,290,291]#label = int(data[1])
                    #if label not in valid_labels:continue #char_tokens = []
                    #label=valid_labels.index(label)#if label not in valid_labels:continue #char_tokens = []
                    char_tokens = []
                    for token in jieba.lcut(data[0][:40]):
                        char_tokens.extend(list(token))
                    # if not use_char:
                    tokens = [token for token in list(text) if token.strip()!='']
                    #tokens = [token for token in jieba.lcut(text) if token.strip()!='']
                    '''
                    tokens = data[0].split('\')
                    # print(tokens)
                    # sys.exit(1)
                    text = ''.join(tokens)
                    # print(tokens)
                    # sys.exit(1)
                    '''
                    # is_short_query_intent = int(data[1]) if len(data)>1 else 0
                    # query_domain = 0#int(data[2])
                    # label = '\t'.join(data[1:])#0 if len(data)>1 and int(data[1]) == 0 else 1#int(data[1])
                    # char_tokens = []
                    # # print(labels)
                    # # if len(labels)==0:
                    # #     print(line.strip().split(","))
                    # #     sys.exit(1)
                    # label = labels[3]
                    # if int(label)>=3:continue

                    # tokens = [token if not token.isdigit() and token.strip()!='' and not token.replace('.','',1).isdigit()else 'digit' for token in tokens]
                # sample = {'raw_text': text, 'is_short': is_short_query_intent,'domain':int(int(query_domain))}
                sample = {'raw_text': text, 'label': label,'item_id':item_id}
                sample2 = {'raw_text': clean_title_str, 'label': label,'item_id':item_id}

                if self.use_bert:
                    sample = self.bert_helper.convert_single_example_to_feature(sample2,self.max_seq_len)
                    sample['raw_text'] = text
                sample['tokens'] = tokens#char_tokens if use_char else [token if not token.isnumeric() and token.strip()!='' and not token.replace('.','',1).isdigit()else 'digit' for token in tokens]
                sample['char_tokens'] = char_tokens
                if self.use_pos_feature:
                    sample['pos_feature'] = token_pos

                # sample={'tokens': jieba.lcut(clean_numbers(data[0])),'raw_text':data[0],'label':label}
                dataset.append(sample)
                y.append(label)
        counter = Counter(y)
        print('.......count of label ..... ')
        print(counter.most_common())
        return dataset

    def convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['token_ids'] = vocab.convert_to_ids(sample['tokens'])
                sample['char_ids'] = vocab.convert_char_to_ids(sample['char_tokens'])
                sample['in_names'] = [int(token in names) for token in sample['tokens']]
                sample['in_ask_words'] =[int(token in ask_word) for token in sample['tokens']]
                found = False if len([True for token in sample['tokens'] if token in names])==0 else True
                # if found: print(' '.join(sample['tokens'])+'\t -----'+' '.join([token for token in sample['tokens'] if token in names]))

    def gen_mini_batches(self, set_name, batch_size, pad_id=0, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id)

    def _one_mini_batch(self, data, indices, pad_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {
            'raw_data': [data[i] for i in indices],
            'token_ids': [],
            'text_len': [],
            'is_short': [],
            'domain':[],
            'features':[],
            'ask_word_feature':[],
            'char_ids':[],
            'char_lens':[],
            'label':[],
            'input_ids':[],
            'input_mask':[],
            'segment_ids':[],
            'bert_text_len':[]

        }
        if self.use_pos_feature:
            batch_data['pos_feature'] = []
        for idx, sample in enumerate(batch_data['raw_data']):
            batch_data['token_ids'].append(sample['token_ids'][:self.max_len])
            if self.use_char:
                batch_data['char_ids'].append(sample['char_ids'][:self.max_c_len])
                batch_data['char_lens'].append(len(batch_data['char_ids'][idx]))
            batch_data['text_len'].append(len(batch_data['token_ids'][idx]))
            if self.use_pos_feature:
                # print(sample)
                # sys.exit(1)
                # if 'pos_feature' not in sample:
                #     print(sample)
                #     sys.exit(1)
                batch_data['pos_feature'].append(sample['pos_feature'][:self.max_len])
            if self.use_name_feature:
                batch_data['features'].append(sample['in_names'][:self.max_len])
                batch_data['ask_word_feature'].append(sample['in_ask_words'][:self.max_len])
            batch_data['label'].append(sample['label'])

            if self.use_bert:
                batch_data['input_ids'].append(sample['input_ids'])
                batch_data['segment_ids'].append(sample['segment_ids'])
                batch_data['input_mask'].append(sample['input_mask'])
                batch_data['use_bert'] = self.use_bert
                batch_data['bert_text_len'].append(sample['bert_text_len'])

            # if 'is_short' in sample:
            #     batch_data['is_short'].append(sample['is_short'])
            # else:
            #     batch_data['is_short'].append(0)
            #     print('error !')
            #     sys.exit(1)
            # if 'domain' in sample:
            #     batch_data['domain'].append(sample['domain'])


        max_len = self.max_len#max(batch_data['text_len'])
        max_c_len = max(batch_data['char_lens'])
        # max_c_len =128
        # print(max_c_len)
        # sys.exit(1)
        for idx, tok_ids in enumerate(batch_data['token_ids']):
            batch_data['token_ids'][idx] = tok_ids + [pad_id] * (max_len - len(tok_ids))
            if self.use_name_feature:
                features = batch_data['features'][idx]
                batch_data['features'][idx] = features+[pad_id]*(max_len-len(tok_ids))
                ask_word = batch_data['ask_word_feature'][idx]
                batch_data['ask_word_feature'][idx] = ask_word+[pad_id]*(max_len-len(tok_ids))
            if self.use_pos_feature:
                pos_feature = batch_data['pos_feature'][idx]
                batch_data['pos_feature'][idx] = pos_feature+[pad_id]*(max_len-len(tok_ids))

        if self.use_char:
            for idx,char_ids in enumerate(batch_data['char_ids']):
                batch_data['char_ids'][idx] = char_ids+[pad_id]*(max_c_len-len(char_ids))

        return batch_data

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['tokens']:
                    yield token


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("brc")
    logger.setLevel(logging.INFO)
    brc_data = DatasetReader('/Users/apple/Downloads/news_qa/news_data_0827/news_data_0827_1w.csv')
    sys.exit(1)
    from data.vocab import Vocab

    vocab = Vocab(lower=True)
    for word in brc_data.word_iter('train'):
        vocab.add(word)

    unfiltered_vocab_size = vocab.size()
    # vocab.filter_tokens_by_cnt(min_cnt=2)

    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.size()))
    brc_data.convert_to_ids(vocab)
    train_batches = brc_data.gen_mini_batches('train', batch_size=16)
    for batch in train_batches:
        print(batch['in'])

        sys.exit(1)
