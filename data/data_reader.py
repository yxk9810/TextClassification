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
from dict.read_dict import *
for name in names: jieba.add_word(name)

class DatasetReader(object):
    def __init__(self, data_file=None, train_file=None, dev_file=None,test_file =None,use_name_feature=True,use_pos_feature=False,use_char=True):
        self.max_len = 22
        self.max_c_len = 40
        self.dataset = []
        self.test_set = []
        self.use_name_feature = use_name_feature
        self.use_pos_feature = use_pos_feature
        self.use_char = use_char
        self.use_splited_token=False

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
            train_writer = open('../dataset/train_yes_no_8k.txt','a+',encoding='utf-8')
            for sample in self.train_set:
                train_writer.write(sample['raw_text']+'\t'+str(sample['label'])+'\n')
            train_writer.close()

            self.dev_set = dataset_shuffled[dev_sample_index:]
            dev_writer = open('../dataset/dev_yes_no_8k.txt', 'a+', encoding='utf-8')
            for sample in self.dev_set:
                dev_writer.write(sample['raw_text'] + '\t' + str(sample['label']) + '\n')
            dev_writer.close()
            sys.exit(1)

        if train_file:
            self.train_set = self._load_dataset(train_file)
        if dev_file:
            self.dev_set = self._load_dataset(dev_file)
        if test_file:
            self.test_set = self._load_dataset(test_file)

        self.logger.info('train set size: {} '.format(len(self.train_set)))
        self.logger.info('dev set size: {}'.format(len(self.dev_set)))

        # self.train_set, self.dev_set, self.test_set = [], [], []
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

    def _load_dataset(self, filename, use_char=False):
        dataset = []
        y = []
        with open(filename, 'r', encoding='utf-8') as lines:
            for line in lines:
                if line.strip() == '' or len(line.strip()) == 0:
                    continue
                if self.use_pos_feature:
                    data = line.strip().split('\t')
                    tokens = [w.split('#pos#')[0] for w in data[0].split()]
                    tokens = [token if not token.isnumeric() and token.strip() != '' and not token.replace('.', '', 1).isdigit() else 'digit' for token in tokens]
                    tokens = [token if re.match('^(\d{1,2}月$|\d{1,2}(日|号)$|\d{4}年$)',token) is None else 'date' for token in  tokens]
                    char_tokens = []
                    for token in tokens:
                        char_tokens.extend(list(token))
                    if use_char:
                        tokens = char_tokens
                    text =''.join(tokens)
                    token_pos = [int(w.split('#pos#')[1]) for w in data[0].split()]
                    if len(data)==1: label = 0
                    else:
                        label = 0 if int(data[1]) == 0 else 1#int(data[1])
                elif self.use_splited_token:
                    data = line.strip().split('\t')
                    tokens = data[1].strip().split(' ')
                    char_tokens = []
                    for token in tokens:
                        char_tokens.extend(list(token))
                    label = int(data[-2].split(' fake: ')[0].strip())
                    text=''.join(data[1].strip().split(' '))

                else:

                    data = line.strip().split('\t')
                    '''
                    back 
                    text  = data[0]
                    char_tokens = []
                    for token in jieba.lcut(clean_numbers(data[0])):
                        char_tokens.extend(list(token))
                    if not use_char:
                        tokens = [token for token in jieba.lcut(clean_date(data[0])) if token.strip()!='']
                    '''
                    tokens = data[0].split('<spliter>')
                    # print(tokens)
                    # sys.exit(1)
                    text = ''.join(tokens)
                    # print(tokens)
                    # sys.exit(1)
                    label = 0 if len(data)>1 and int(data[1]) == 0 else 1#int(data[1])
                    char_tokens = []
                    # tokens = [token if not token.isdigit() and token.strip()!='' and not token.replace('.','',1).isdigit()else 'digit' for token in tokens]
                sample = {'raw_text': text, 'label': label}

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
            'labels': [],
            'features':[],
            'ask_word_feature':[],
            'char_ids':[],
            'char_lens':[]

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
            if 'label' in sample:
                batch_data['labels'].append(sample['label'])
            else:
                batch_data['labels'].append(0)
                print('error !')
                sys.exit(1)
        max_len = max(batch_data['text_len'])
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
    brc_data = DatasetReader('/Users/apple/Downloads/news_qa/intent_yes_no_all.txt')
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
        print(batch['features'])

        sys.exit(1)
