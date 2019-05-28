import os
import tensorflow as tf
import abc
import numpy as np
from typing import List, Iterator, Iterable
from random import shuffle
import time
from tqdm import tqdm

from modules.feeder import DataFeeder
from modules.static_dict import CRF_VOCA, REL2ID
from modules.misc import Pre_Misc, Post_Misc, Common_Misc
from modules.post import EvaluatePrediction, post_process

class BasicFrame:
    """
    tensorflow 模型框架的基础类
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, sess = None, config = None, logger = None, summary = True, summary_dir = './log'):
        self.sess = sess
        self.best_score = 0
        self.config = config
        self.model_name = self.config.nn_model
        self.logger = logger
        try:
            self.nn = getattr(__import__('NN_models'), self.model_name)(config=config)
        except AttributeError:
            raise NotImplementedError('can not find {}'.format(self.model_name))
        self.nstep_summary = 0
        self.summary = summary
        self.summary_dir = summary_dir
        self.sess.run(tf.global_variables_initializer())
        if logger is not None:
            print_config = '\n'.join(['nn_model: {}'.format(self.config.nn_model),
                                      'learning_rate: {}'.format(self.config.learning_rate),
                                      'use_pretrain_embedding: {}'.format(self.config.use_pretrain_embedding),
                                      'use_crf: {}'.format(self.config.use_crf)])
            self.logger.info('\n' + print_config)

    def save(self, name, var_list=None, save_dir='./save'):
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, '{}.ckpt'.format(name))
        saver = tf.train.Saver(var_list=var_list)
        saver.save(self.sess, checkpoint_path, global_step=None)  # save model

    def restore(self, name, var_list=None, save_dir='./save'):
        checkpoint_path = os.path.join(save_dir, '{}.ckpt'.format(name))
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(self.sess, checkpoint_path)

    def create_batch_generator(self,
                               X,
                               batch_size,
                               nepoch,
                               shuffle_bool=False):

        if isinstance(X, Iterator):
            iter_ = X
            onebatch = []
            n = 0
            for inst in iter_:
                onebatch.append(inst)
                n += 1
                if n == batch_size:
                    yield onebatch
                    onebatch = []
                    n = 0
            if n > 0:
                yield onebatch
        elif isinstance(X, List):
            n=0; i=0; k=0
            list_length = len(X)
            onebatch = []
            while True:
                if k >= list_length:
                    i += 1
                    if i >= nepoch:
                        break
                    if shuffle_bool:
                        shuffle(X)
                    k = 0
                onebatch.append(X[k])
                k+=1
                n+=1
                if n == batch_size:
                    yield onebatch
                    onebatch = []
                    n = 0
            if n > 0:
                yield onebatch
        else:
            raise NotImplementedError

    @abc.abstractmethod
    def make_one_batch(self, *arg, **kwargs):
        raise NotImplementedError
    @abc.abstractmethod
    def fit(self, *arg, **kwargs):
        raise NotImplementedError
    @abc.abstractmethod
    def predict_proba(self, *arg, **kwargs):
        raise NotImplementedError
    @abc.abstractmethod
    def predict(self, *arg, **kwargs):
        raise NotImplementedError
    @abc.abstractmethod
    def score(self, *arg, **kwargs):
        raise NotImplementedError


class JointFrame(BasicFrame):

    def make_one_batch(self, insts, max_length = None, is_testing = False):
        self.ntime_divide_by_two = 0
        if 'unet' in self.model_name:
            self.ntime_divide_by_two = 0  # 如果使用unet，max_sentence_length必须是2**5的倍数

        batch_size = len(insts)
        has_sub_label = 'sub_label' in insts[0]
        has_simple_label = 'spo_list' in insts[0]
        has_crf_label = 'crf_label' in insts[0]
        use_bert = self.config.use_bert

        if max_length is None:  # if no max_length is given, set it to be the max length in the batch.
            max_length = 1
            for inst in insts:
                if inst['char_size'] > max_length:
                    max_length = inst['char_size']

        denominator = 2**self.ntime_divide_by_two
        max_length = max_length + denominator - max_length % denominator # so that max_length % denominator == 0

        char_array = np.zeros(shape=(batch_size, max_length))
        bert_char_array = np.zeros(shape=(batch_size, max_length))
        bert_mask_array = np.zeros(shape=(batch_size, max_length))
        mask_array = np.zeros(shape=batch_size)
        pos_array = np.zeros(shape=(batch_size, max_length, 2))
        seg_array = np.zeros(shape=(batch_size, max_length, 2))
        ne_array = np.zeros(shape=(batch_size, max_length, 1))
        par_array = np.zeros(shape=(batch_size, max_length, 1))
        freq_array = np.zeros(shape=(batch_size, max_length, 1))

        ner_label_array = np.zeros(shape=(batch_size, max_length, 2), dtype=np.float)
        ner_weight_array = np.zeros(shape=(batch_size, max_length, 2))

        simple_array = np.zeros(shape=(batch_size, max_length, self.config.n_relation*2), dtype=np.float)
        classify_label_array = np.zeros(shape=(batch_size, self.config.n_relation), dtype=np.float)

        crf_label_array = []
        rel_indices = []
        crf_mask_array = []

        for k, inst in enumerate(insts):
            char_size = inst['char_size']
            char_array[k, :char_size] = np.array(inst['char_index'])
            mask_array[k] = char_size
            pos_array[k, :char_size, 0] = np.array(inst['pos_index'])
            pos_array[k, :char_size, 1] = np.array(inst['ltp_pos_index'])
            seg_array[k, :char_size, 0] = np.array(inst['bmes_index'])
            seg_array[k, :char_size, 1] = np.array(inst['ltp_bmes_index'])
            ne_array[k, :char_size, 0] = np.array(inst['ltp_ner_index'])
            par_array[k, :char_size, 0] = np.array(inst['ltp_par_index'])
            freq_array[k, :char_size, 0] = np.array(inst['char_freq'])
            if has_sub_label:
                ner_label_array[k, :char_size, 0] = np.array(inst['sub_label'])
                ner_label_array[k, :char_size, 1] = np.array(inst['ob_label'])
                ner_weight_array[k, :char_size, 0] = np.array(inst['sub_weight'])
                ner_weight_array[k, :char_size, 1] = np.array(inst['ob_weight'])

            if has_simple_label:
                text = inst['raw_text']
                for spo in inst['spo_list']:
                    rel = '-'.join([spo['predicate'], spo['subject_type'], spo['object_type']])
                    rel_id = REL2ID[rel]
                    for span in Common_Misc.my_finditer(spo['subject'], text):
                        simple_array[k, span[0]:span[1], rel_id] = 1
                    for span in Common_Misc.my_finditer(spo['object'], text):
                        simple_array[k, span[0]:span[1], rel_id + self.config.n_relation] = 1

            if has_crf_label and not is_testing:
                for key in inst['crf_label']:
                    arow = [CRF_VOCA[inst['crf_label'][key][ii]] if ii < char_size else 0 for ii in range(max_length)]
                    crf_label_array.append(arow)
                    rel_indices.append([k, REL2ID[key]])
                    classify_label_array[k, REL2ID[key]] = 1
                    crf_mask_array.append(char_size)
            if is_testing:
                for kk in range(self.config.n_relation):
                    crf_label_array.append([0 for _ in range(max_length)])
                    rel_indices.append([k, kk])
                    crf_mask_array.append(char_size)
            if use_bert:
                bert_size = len(inst['bert_char_index'])
                assert bert_size == char_size
                bert_char_array[k, :bert_size] = np.array(inst['bert_char_index'])
                bert_mask_array[k, :bert_size] = 1

        feed_dict = {self.nn.text: char_array,
                     self.nn.mask: mask_array,
                     self.nn.postag: pos_array,
                     self.nn.bmes: seg_array,
                     self.nn.netag: ne_array,
                     self.nn.freq: freq_array}
        feed_dict.update({self.nn.is_training: False})

        if has_sub_label:
            feed_dict.update({self.nn.ner_label: ner_label_array,
                              self.nn.ner_weight: ner_weight_array})
        if has_simple_label:
            feed_dict.update({self.nn.simple_label: simple_array})

        if has_crf_label:
            feed_dict.update({self.nn.crf_label: np.array(crf_label_array),
                              self.nn.rel_indices: np.array(rel_indices),
                              self.nn.crf_mask: np.array(crf_mask_array),
                              self.nn.classify_label: np.array(classify_label_array)})
        if is_testing:
            feed_dict.update({self.nn.crf_label: np.array(crf_label_array),
                              self.nn.rel_indices: np.array(rel_indices),
                              self.nn.crf_mask: np.array(crf_mask_array)})
        if use_bert:
            feed_dict.update({self.nn.bert_index: bert_char_array,
                              self.nn.bert_mask: bert_mask_array})

        return feed_dict

    def fit(self, X, dev_X, batch_size=None, nepoch=None, save_best=False, save_dir='./save',
            loss_function=None, balance_class = False):
        self.logger.info('Training...')
        if batch_size is None:
            batch_size = self.config.batch_size
        else:
            self.config.batch_size = batch_size
        if nepoch is None:
            nepoch = self.config.nepoch
        if self.summary:
            if os.path.exists(self.summary_dir):
                os.system('rm -rf ' + self.summary_dir)
            self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)

        self.current_epoch = 0
        self.current_batch = 0
        if isinstance(X, DataFeeder):
            self.n_train = X.count()
            self.train_iter = self.create_batch_generator(
                X.iterator(nepoch=nepoch, shuffle_bool=True, balance_class=balance_class),
                batch_size=batch_size,
                nepoch=nepoch)
        elif isinstance(X, List):
            self.n_train = len(X)
            self.train_iter = self.create_batch_generator(X, batch_size=batch_size, nepoch=nepoch)
        else:
            raise NotImplementedError
        self.logger.info('total train sample number: {}'.format(self.n_train))


        starttime = time.time()
        for onebatch in self.train_iter:
            fedict = self.make_one_batch(onebatch, max_length=None)
            if self.current_batch * batch_size >= self.current_epoch * self.n_train:
                self.current_epoch += 1
                self.logger.info("\n## The {} Epoch, All {} Epochs ! ##".format(self.current_epoch, nepoch))
                loss = self.sess.run(self.nn.loss, feed_dict=fedict)
                self.logger.debug("\nbatch_count = {}, loss is {:.6f}.".format(self.current_batch, loss))
                endtime = time.time()
                self.logger.debug("\nTrain Time {:.3f}".format(endtime - starttime))
                starttime = time.time()

            if (dev_X is not None) and (self.current_batch % 100 == 10):
                self.logger.info('On dev set:')
                f1 = self.score(dev_X, word_level=True)
                if save_best:
                    if f1 > self.best_score:
                        save_name = self.model_name + '-' + str(self.current_epoch)
                        self.save(save_name, save_dir=save_dir)
                        self.best_score = f1
                        self.logger.info('new best model (f1:{}) saved as : {}'.format(self.best_score, save_name))
                if self.current_batch % 1000 == 910:
                    save_name = self.model_name + ('_latest_%.3f' % f1)
                    self.save(save_name, save_dir=save_dir)

            self.current_batch += 1

            feed_dict = {self.nn.step: self.current_batch, self.nn.is_training: True}
            feed_dict.update(fedict)
            if loss_function is None:
                loss, _ = self.sess.run([self.nn.loss, self.nn.train_step], feed_dict=feed_dict)
            else:
                train_step = self.nn.optimizer.minimize(loss_function)
                loss, _ = self.sess.run([loss_function, train_step], feed_dict=feed_dict)
            if self.current_batch % 5 == 1:
                self.logger.debug("\nbatch_count = {}, loss is {:.6f}.".format(self.current_batch, loss))
            if self.summary:
                stat = self.sess.run(self.nn.stat, feed_dict=feed_dict)
                self.summary_writer.add_summary(stat, self.current_batch)

        return None

    def predict_proba(self, X, is_testing=False):
        preds = []; probas = []
        for onebatch in tqdm(self.create_batch_generator(X, batch_size=self.config.batch_size*2, nepoch=1)):
            fedict = self.make_one_batch(onebatch, max_length= self.config.max_sentence_length, is_testing=is_testing)
            # bpred, bproba = self.sess.run([self.nn.crf_predict, self.nn.crf_predict_proba], feed_dict=fedict)
            try:
                bpred, bproba = self.sess.run([self.nn.crf_predict, self.nn.predict_proba], feed_dict=fedict)
            except:
                bpred, bproba = self.sess.run([self.nn.crf_predict, self.nn.extra_crf_proba], feed_dict=fedict)
            #
            preds += list(bpred)
            probas += list(bproba)
        return preds, probas

    # def simple_proba(self, X):
    #     probas = []
    #     for onebatch in tqdm(self.create_batch_generator(X, batch_size=256, nepoch=1)):
    #         fedict = self.make_one_batch(onebatch, max_length= self.config.max_sentence_length)
    #         proba = self.sess.run(self.nn.simple_proba, feed_dict=fedict)
    #         probas += list(proba)
    #     return probas

    def predict(self, X):
        preds, probas = self.predict_proba(X)
        pred_insts = self._transform_submit_form(preds=preds, insts=X)
        return pred_insts

    def score(self, X, word_level = False, show_worst = False):
        origin_is_train = self.config.is_training
        self.config.is_training = False
        preds, probas = self.predict_proba(X)
        self.config.is_training = origin_is_train
        golds = []
        for inst in X:
            for key in inst['crf_label']:
                golds.append(np.array([CRF_VOCA[tag] for tag in inst['crf_label'][key]]))

        ncorrect = 0
        nreal = 0
        npred = 0
        for pred, gold in zip(preds, golds):
            pred = np.array(pred[:len(gold)])
            ncorrect += np.sum((np.array(pred) == gold)*(gold > 0))
            nreal += np.sum(gold > 0)
            npred += np.sum(pred > 0)
        precision = ncorrect/(npred + 1e-9)
        recall = ncorrect/(nreal + 1e-9)
        f1 = 2 * ncorrect/(npred + nreal + 1e-9)

        if word_level:
            preds_submit_format = self._transform_submit_form(preds=preds, insts=X, probas=probas)
            evaluater = EvaluatePrediction(logger=self.logger)
            evaluater.evluate(preds=preds_submit_format, golds=X, show_worst=show_worst)

        self.logger.info('Average recall: {}'.format(recall))
        self.logger.info('Average precision: {}'.format(precision))
        self.logger.info('Average f1_score: {}'.format(f1))

        return f1

    def evaluate(self, X, is_testing = True, show_worst = False, n_expect = 2.1):
        preds, probas = self.predict_proba(X, is_testing=is_testing)
        pred_insts = self._transform_submit_form(preds=preds, insts=X, probas=probas, is_testing=is_testing)
        pred_insts = Post_Misc.sort_and_cut(pred_insts, n_expect)
        f1 = 0
        if 'spo_list' in X[0]:
            evaluater = EvaluatePrediction(logger=self.logger)
            f1 = evaluater.evluate(preds=pred_insts, golds=X, show_worst=show_worst)
        return f1, pred_insts

    def gene_proba_feature(self, X, is_testing = True):
        preds, probas = self.predict_proba(X, is_testing=is_testing)
        CRF_ID2TAG = list(CRF_VOCA.keys())
        crftags = [[CRF_ID2TAG[k] for k in line] for line in preds]

        ii = 0
        all_spos = []
        for inst in X:
            if is_testing:
                rels = sorted(list(REL2ID.keys()), key=lambda x: REL2ID[x])
            else:
                rels = inst['crf_label']
            for key in rels:
                subjects_span_dict = {}
                objects_span_dict = {}
                pred_word = Post_Misc.crftag2word(crftags[ii][:inst['char_size']], inst['raw_text'])
                for tup in pred_word:
                    if tup[0] == 'sub':
                        if tup[1] in subjects_span_dict:
                            subjects_span_dict[tup[1]].append(tup[2])
                        else:
                            subjects_span_dict[tup[1]] = [tup[2]]
                    else:
                        if tup[1] in objects_span_dict:
                            objects_span_dict[tup[1]].append(tup[2])
                        else:
                            objects_span_dict[tup[1]] = [tup[2]]
                pairs = Post_Misc.match_multi_pair(subjects_span_dict, objects_span_dict)  # 处理多配对
                for pair in pairs:
                    spo = {'rel': key,
                           'subject': pair[0],
                           'object': pair[1],
                           'subspans': subjects_span_dict[pair[0]],
                           'obspans': objects_span_dict[pair[1]],
                           'inst_id': inst['_id'],
                           'proba_array': probas[ii][:inst['char_size']].tolist()}
                    all_spos.append(spo)
                ii += 1
        return all_spos


    def pred_spos_with_proba_array(self, X, is_testing = True):
        preds, probas = self.predict_proba(X, is_testing=is_testing)
        CRF_ID2TAG = list(CRF_VOCA.keys())
        crftags = [[CRF_ID2TAG[k] for k in line] for line in preds]
        ii = 0
        all_spos = []
        for inst in X:
            text = inst['raw_text']
            if is_testing:
                rels = sorted(list(REL2ID.keys()), key=lambda x: REL2ID[x])
            else:
                rels = inst['crf_label']
            for key in rels:
                subjects_span_dict = {}
                objects_span_dict = {}
                pred_word = Post_Misc.crftag2word(crftags[ii][:inst['char_size']], inst['raw_text'])
                for tup in pred_word:
                    if tup[0] == 'sub':
                        if tup[1] in subjects_span_dict:
                            subjects_span_dict[tup[1]].append(tup[2])
                        else:
                            subjects_span_dict[tup[1]] = [tup[2]]
                    else:
                        if tup[1] in objects_span_dict:
                            objects_span_dict[tup[1]].append(tup[2])
                        else:
                            objects_span_dict[tup[1]] = [tup[2]]
                pairs = Post_Misc.match_multi_pair(subjects_span_dict, objects_span_dict)  # 处理多配对
                for pair in pairs:
                    spo = {'rel': key,
                           'subject': pair[0],
                           'object': pair[1],
                           'subspans': subjects_span_dict[pair[0]],
                           'obspans': objects_span_dict[pair[1]],
                           'inst_id': inst['_id'],
                           'text': text,
                           'proba_array': probas[ii][:inst['char_size']].tolist()}
                    all_spos.append(spo)
                ii += 1
        return all_spos




    def _transform_submit_form(self, preds, insts, probas, is_testing=False):

        CRF_ID2TAG = list(CRF_VOCA.keys())
        crftags = [[CRF_ID2TAG[k] for k in line] for line in preds]

        ii = 0
        pred_insts = []
        for inst in insts:
            if len(inst['spo_list']) > 4:
                debug=0 # debug
            pred_inst = {'_id': inst['_id'], 'text': inst['raw_text'], 'spo_list': []}
            if is_testing:
                rels = sorted(list(REL2ID.keys()), key=lambda x: REL2ID[x])
            else:
                rels = inst['crf_label']
            for key in rels:

                predicate, subject_type, object_type = key.split('-')
                subjects_span_dict = {}
                objects_span_dict = {}
                pred_word = Post_Misc.crftag2word(crftags[ii][:inst['char_size']], inst['raw_text'])
                for tup in pred_word:
                    if tup[0] == 'sub':
                        if tup[1] in subjects_span_dict:
                            subjects_span_dict[tup[1]].append(tup[2])
                        else:
                            subjects_span_dict[tup[1]] = [tup[2]]
                    else:
                        if tup[1] in objects_span_dict:
                            objects_span_dict[tup[1]].append(tup[2])
                        else:
                            objects_span_dict[tup[1]] = [tup[2]]
                pairs = Post_Misc.match_multi_pair(subjects_span_dict, objects_span_dict)  # 处理多配对
                spos_with_proba = []
                for pair in pairs:
                    spo = {'predicate': predicate,
                           'subject_type': subject_type,
                           'object_type': object_type,
                           'subject': pair[0],
                           'object': pair[1],
                           'proba': Post_Misc.calculate_proba(subspans=subjects_span_dict[pair[0]],
                                                              obspans=objects_span_dict[pair[1]],
                                                              proba=probas[ii][:inst['char_size']]),
                           'subspans': subjects_span_dict[pair[0]],
                           'obspans': objects_span_dict[pair[1]]}
                    spos_with_proba.append(spo)
                pred_inst['spo_list'] += spos_with_proba
                ii += 1

            pred_inst = post_process(pred_inst)
            pred_insts.append(pred_inst)

        return pred_insts


