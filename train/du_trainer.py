#coding:utf-8
import tensorflow as tf
from collections import defaultdict
import os
import tensorflow as tf
import logging
import os
import numpy as np
from collections import defaultdict
import sys
# from dict.read_dict import movie_name
import time
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

movie_name = set()
# from data.elmo_data_handler import *
class Trainer(object):
    def __init__(self):
        pass

    @staticmethod
    def _train_sess(model, batches, summary_writer, save_summary_steps):
        global_step = tf.train.get_or_create_global_step()
        i =0
        for batch in batches:
            i+=1
            train_batch = {}
            train_batch['token_ids'] = batch['token_ids']
            train_batch['labels'] = batch['label']
            # train_batch['domain'] = batch['domain']
            train_batch['text_len'] = batch['text_len']
            train_batch['features'] = batch['features']
            if 'use_bert' in batch:
                train_batch['input_ids'] = batch['input_ids']
                train_batch['input_mask'] = batch['input_mask']
                train_batch['segment_ids'] = batch['segment_ids']
                train_batch['text_len'] = batch['bert_text_len']

            text_tokenized = [d['tokens'] for d in batch['raw_data']]
            # train_batch['elmo_token_ids'] = batcher.batch_sentences(text_tokenized)
            # train_batch['soft_target'] = batch['soft_target']
            # import numpy as np
            # print(np.array(train_batch['soft_target']).shape)
            # sys.exit(1)
            # train_batch['ask_word_feature']=batch['ask_word_feature']
            # train_batch['pos_feature'] = batch['pos_feature']
            # train_batch['char_lens'] = batch['char_lens']
            train_batch['char_ids'] = batch['char_ids']
            train_batch["training"] = True
            feed_dict = {ph: train_batch[key] for key, ph in model.input_placeholder_dict.items()}
            if i % save_summary_steps == 0:
                _, _, loss_val, summ, global_step_val = model.session.run([model.train_op, model.train_update_metrics,
                                                                           model.loss, model.summary_op, global_step],
                                                                          feed_dict=feed_dict)
                if summary_writer is not None:
                    summary_writer.add_summary(summ, global_step_val)
            else:
                _, _, loss_val = model.session.run([model.train_op, model.train_update_metrics, model.loss],
                                                   feed_dict=feed_dict)
            #break

        metrics_values = {k: v[0] for k, v in model.train_metrics.items()}
        metrics_val = model.session.run(metrics_values)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
        logging.info("- Train metrics: " + metrics_string)

    @staticmethod
    def _eval_sess(model, batches, summary_writer):
        global_step = tf.train.get_or_create_global_step()
        final_output = defaultdict(list)
        labels = []
        domain = []
        text = []
        for idx,batch in enumerate(batches):
            eval_batch = {}
            eval_batch['token_ids'] = batch['token_ids']
            # eval_batch['domain'] = batch['domain']
            eval_batch['labels'] = batch['label']
            eval_batch['text_len'] = batch['text_len']
            eval_batch['char_lens'] = batch['char_lens']
            # eval_batch['features'] = batch['features']
            # eval_batch['soft_target'] = batch['soft_target']
            #
            # # eval_batch['ask_word_feature'] = batch['ask_word_feature']
            eval_batch['char_ids'] =batch['char_ids']
            if 'use_bert' in batch:
                eval_batch['input_ids'] = batch['input_ids']
                eval_batch['input_mask'] = batch['input_mask']
                eval_batch['segment_ids'] = batch['segment_ids']
                eval_batch['text_len'] = batch['bert_text_len']

            #
            # eval_batch['pos_feature'] = batch['pos_feature']

            labels.extend(batch['label'])
            # domain.extend(batch['domain'])
            for sample in batch['raw_data']:
                # print(sample['raw_text'])
                # sys.exit(1)
                text.append(sample['raw_text'])
            # sys.exit(1)
            eval_batch["training"] = False
            feed_dict = {ph: eval_batch[key] for key, ph in model.input_placeholder_dict.items()}
            _, output,training = model.session.run([model.eval_update_metrics, model.output_variable_dict,model.training], feed_dict=feed_dict)
            for key in output.keys():
                final_output[key] += [v for v in output[key]]

        # Get the values of the metrics
        metrics_values = {k: v[0] for k, v in model.eval_metrics.items()}
        metrics_val = model.session.run(metrics_values)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
        logging.info("- Eval metrics: " + metrics_string)

        # Add summaries manually to writer at global_step_val
        if summary_writer is not None:
            global_step_val = model.session.run(global_step)
            for tag, val in metrics_val.items():
                summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
                summary_writer.add_summary(summ, global_step_val)

        return final_output,labels,text,domain

    @staticmethod
    def inference(model, batch_generator, steps):
        final_output = defaultdict(list)
        for _ in range(steps):
            batch = batch_generator.next()
            batch['training'] = False
            feed_dict = {ph: batch[key] for key, ph in model.input_placeholder_dict.items()}
            output = model.session.run(model.output_variable_dict, feed_dict=feed_dict)
            for key in output.keys():
                final_output[key] += [v for v in output[key]]
        return final_output

    @staticmethod
    def _train_and_evaluate(model,brc_reader,evaluator, epochs=1, eposides=1,batch_size=32,
                            save_dir=None, summary_dir=None, save_summary_steps=10):
        best_saver = tf.train.Saver(max_to_keep=1) if save_dir is not None else None
        train_summary = tf.summary.FileWriter(os.path.join(summary_dir, 'train_summaries')) if summary_dir else None
        eval_summary = tf.summary.FileWriter(os.path.join(summary_dir, 'eval_summaries')) if summary_dir else None
        pad_id = 0 

        best_eval_acc= 0.0
        for epoch in range(epochs):
            logging.info("Epoch {}/{}".format(epoch + 1, epochs))
            model.session.run(model.train_metric_init_op)
            # one epoch consists of several eposides
            train_batches = brc_reader.gen_mini_batches('train', batch_size, pad_id, shuffle=True)

            Trainer._train_sess(model, train_batches, train_summary, save_summary_steps)

            # Save weights
            if save_dir is not None:
                last_save_path = os.path.join(save_dir, 'last_weights', 'after-eposide')
                model.save(last_save_path, global_step=1)

            # Evaluate for one epoch on dev set

            model.session.run(model.eval_metric_init_op)

            eval_batches = brc_reader.gen_mini_batches('dev',batch_size,pad_id,shuffle=False)

            output,labels,text,domains= Trainer._eval_sess(model, eval_batches, eval_summary)
            score = Trainer.get_score(output,labels)
            #
            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in score.items())
            logging.info("- Eval metrics: " + metrics_string)

            # # Save best weights
            if  best_eval_acc < score['f1']:
                logging.info('save ......')
                filename = 'dev_pred.txt'
                best_eval_acc= score['acc']
                if os.path.exists(filename):
                    os.remove(filename)
                writer = open(filename,'a+',encoding='utf-8')
                import sys
                for index in range(len(labels)):
                    if int(output['predict'][index])!=int(labels[index]):
                        found_names = [name  for name in movie_name if name in text[index]]
                        if len(found_names)>0:
                            max_len = max([len(name) for name in found_names])
                            #print(text[index]+'\t'+' '.join([name for name in found_names if len(name)==max_len])+str(output['predict'][index])+'\t'+str(labels[index])+'\n')
                for index in range(len(labels)):
                    writer.write(text[index]+'\t'+str(output['predict'][index])+'\t'+str(labels[index])+'\n')
                writer.close()

                pad_id = 0
                infer_batch_size = 64
                eval_batches = brc_reader.gen_mini_batches('test',64 , pad_id, shuffle=False)

                # sys.exit(1)
                #sys.exit(1)
            #     best_eval_score = eval_score
                best_save_path = os.path.join(save_dir, 'best_weights', 'after-eposide')
                best_save_path = best_saver.save(model.session, best_save_path, global_step=epoch)
                logging.info("- Found new best model, saving in {}".format(best_eval_acc))
                Trainer._test_sess(model, eval_batches,best_eval_acc)
            else:
                break#sys.exit(1)

    @staticmethod
    def get_score(output,labels):
        train_probs = [prob[1] for prob in output['prob']]
        # from utils.proprocess import  bestThresshold
        # delta,best_f1 = bestThresshold(labels,train_probs)
        # print('--------predict with threshold---------')
        # threshold_pred = [1 if prob>delta else 0 for prob in train_probs]
        # print(classification_report(labels,threshold_pred,digits=4))
        # print('--------------end threshold ------------')

        # for prob in output['prob']:
        #     print(prob)
        #sys.exit(1)
        pred = output['predict']
        from sklearn.metrics import f1_score
        data = classification_report(labels,pred,digits=4)
        print(data)
        f1 = data
        # print(data.split('\n')[3].split('    '))
        f1 = float(data.split('\n')[3].split('    ')[-3])
        # sys.exit(1)
        # f1=0.0
        acc = np.sum(np.equal(np.array(pred), np.array(labels))) / float(len(labels))

        return {'acc':acc,'f1':acc}
    @staticmethod
    def _evaluate(model, batch_generator, evaluator):
        # Evaluate for one epoch on dev set
        batch_generator.init()
        model.session.run(model.eval_metric_init_op)
        eval_instances = batch_generator.get_instances()

        eval_num_steps = (len(
            eval_instances) + batch_generator.get_batch_size() - 1) // batch_generator.get_batch_size()
        output = Trainer._eval_sess(model, batch_generator, eval_num_steps, None)
        pred_answer = model.get_best_answer(output, eval_instances)
        score = evaluator.get_score(pred_answer)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in score.items())
        logging.info("- Eval metrics: " + metrics_string)

    @staticmethod
    def _inference(model, batch_generator):
        batch_generator.init()
        model.session.run(model.eval_metric_init_op)
        instances = batch_generator.get_instances()
        eval_num_steps = (len(instances) + batch_generator.get_batch_size() - 1) // batch_generator.get_batch_size()
        output = Trainer.inference(model, batch_generator, eval_num_steps)
        pred_answers = model.get_best_answer(output, instances)
        return pred_answers
    @staticmethod
    def _test_sess(model, batches, f1='inference'):
        final_output = defaultdict(list)
        samples = []
        for batch in batches:
            test_batch = {}
            start = time.time()#test_batch = {}
            for sample in batch['raw_data']:
                samples.append(sample)
            test_batch['token_ids'] = batch['token_ids']
            test_batch['text_len'] = batch['text_len']
            test_batch['char_lens'] = batch['char_lens']
            test_batch['features'] = batch['features']
            if 'use_bert' in batch:
                test_batch['input_ids'] = batch['input_ids']
                test_batch['input_mask'] = batch['input_mask']
                test_batch['segment_ids'] = batch['segment_ids']
                test_batch['text_len'] = batch['bert_text_len']
            # test_batch['pos_feature'] = batch['pos_feature']

            # eval_batch['ask_word_feature'] = batch['ask_word_feature']
            test_batch['char_ids'] = batch['char_ids']

            test_batch["training"] = False
            feed_dict = {ph: test_batch[key] for key, ph in model.input_placeholder_dict.items() if
                         key not in ['labels','domain']}
            output, training = model.session.run(
                [model.output_variable_dict, model.training], feed_dict=feed_dict)
            for key in output.keys():
                final_output[key] += [v for v in output[key]]
            #print("cost ..."+str(time.time()-start))#test_batch = {}
        # Get the values of the metrics
        print(len(samples))# Get the values of the metrics
        print(len(final_output['predict']))# Get the values of the metrics
        #sys.exit(1)#print(len(final_output['predict']))# Get the values of the metrics
        writer = open(str(f1)+'_1011.txt','a+',encoding='utf-8')
        labels = [] 
        predict_labels = [] 
        for i,sample in enumerate(samples):
            labels.append(sample['label'])
            predict_labels.append(final_output['predict'][i])
            writer.write(sample['raw_text']+'\t'+str(final_output['predict'][i])+'\t'+' '.join([str(v) for v in final_output['prob'][i]])+'\n')
        writer.close()
        print(classification_report(labels,pred,digits=4))
        return final_output, samples
