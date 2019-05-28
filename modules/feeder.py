from pymongo import MongoClient
from tqdm import tqdm
from random import shuffle
import numpy as np
from typing import List, Iterable
import os
import json


class DataFeeder():
    """
    储存、操作样本索引的类
    """

    ORIGIN_TRAIN_NUM = 173109
    ORIGIN_DEV_NUM = 21639
    ORIGIN_TEST_NUM = 9949

    def __init__(self, dataset = 'dev_data', datatype = 'mongo', label_key = 'rels', output_dir = './data/processed_data/',
                 use_relabel_data = False, relabel_dataset = 'train_relabel', use_augment_data = False):

        self.name = dataset
        self.class_id = {}
        self.ignore_id = []
        self.label_key = label_key
        self._num = 0
        self.output_dir = output_dir
        self.class_id_path = os.path.join(output_dir, '{}_{}.json'.format(self.name.split('_')[0], label_key))
        self.use_relabel_data = use_relabel_data
        self.relabel_dataset = relabel_dataset

        if datatype == 'mongo':
            mongo_collection_name = dataset
            self.agent = MongoAgent(collection_name=mongo_collection_name)
            self._num = self.agent.collection.count_documents({})
            if 'test' not in dataset:
                self.get_class_id()
            if self.use_relabel_data:
                self.relabel_agent = MongoAgent(collection_name=self.relabel_dataset)
            if not use_augment_data:
                self._num = min(DataFeeder.ORIGIN_TRAIN_NUM, self._num)
                for key in self.class_id:
                    self.class_id[key] = [_id for _id in self.class_id[key] if _id < DataFeeder.ORIGIN_TRAIN_NUM]
        else:
            raise NotImplementedError

    def count(self):
        return self._num

    def iterator(self,
                 nsample=None,
                 nepoch = 1,
                 shuffle_bool=False,
                 balance_class=False,
                 minimal_class_nsample=0):

        IDs = list(range(self._num))
        if shuffle_bool:
            shuffle(IDs)

        if nsample is None:
            total_num = self._num * nepoch
        else:
            total_num = nsample

        if not balance_class:
            _nepoch = int(np.ceil(total_num/self._num))
            total_ids = (IDs * _nepoch)[:total_num]
            for _id in total_ids:
                yield self.load_by_id(_id)
        else:
            ii = 0
            turn = 0
            key_list = [key for key in self.class_id.keys() if len(self.class_id[key]) > minimal_class_nsample]
            while ii < total_num:
                for key in key_list:
                    _id = self.class_id[key][turn % len(self.class_id[key])]
                    sample = self.load_by_id(Id=_id)
                    yield sample
                    ii += 1
                turn += 1
                shuffle(key_list)
                if ii % self._num == 1:
                    for key in key_list:
                        shuffle(self.class_id[key])

        #     ii = 0
        #     turn = 0
        #     total_ids = []
        #     key_list = [key for key in self.class_id.keys() if len(self.class_id[key]) > minimal_class_nsample]
        #     for key in key_list:
        #         shuffle(self.class_id[key])
        #     while ii < total_num:
        #         for key in key_list:
        #             _id = self.class_id[key][turn % len(self.class_id[key])]
        #             total_ids.append(_id)
        #             ii += 1
        #             if ii >= total_num:
        #                 break
        #         turn += 1
        #         shuffle(key_list)
        # for ID in total_ids:
        #     sample = self.load_by_id(Id=ID)
        #     yield sample

    def load_by_id(self, Id):
        if isinstance(Id, int):
            res = self.agent.load_by_id(Id)
            if self.use_relabel_data:
                relabel = self.relabel_agent.load_by_id(Id)
                if relabel is not None:
                    res['spo_list'] = relabel['spo_list'] # 如果在修正数据集里，修正label
        elif isinstance(Id, Iterable):
            def generator_(Id):
                for _id in Id:
                    yield self.load_by_id(_id)
            res = generator_(Id)
        else:
            raise NotImplementedError
        return res

    def get_class_id(self, remake=False):
        class_id_path = self.class_id_path
        if (not remake) and os.path.exists(class_id_path):
            print('class_id exists. Loading from: {}'.format(class_id_path))
            self._load_class_id(class_id_path)
        else:
            print('get the ids for each class based on: ', self.label_key)
            dct = {}
            for sample in tqdm(self.iterator()):
                classes = sample[self.label_key]
                for class_ in classes:
                    if class_ in dct:
                        dct[class_].append(sample['_id'])
                    else:
                        dct[class_] = [sample['_id']]
            self.class_id = dct
            self._save_class_id(class_id_path)
            print('Getting class id finished. Saved at: {}'.format(class_id_path))

    def _load_class_id(self, path):
        with open(path) as f:
            self.class_id = json.load(f)

    def _save_class_id(self, path):
        with open(path, 'w') as f:
            json.dump(self.class_id, f)

    def augment_rare_class(self, less_than, nmulti = 1, logger = None):
        """
        对样本数较少的类做数据扩增，扩增n倍
        :return:
        """
        new_id = DataFeeder.ORIGIN_TRAIN_NUM
        for class_ in self.class_id:
            indices = [_id for _id in self.class_id[class_] if _id < DataFeeder.ORIGIN_TRAIN_NUM]  # 只用原始数据进行扩增
            class_count = len(indices)
            if class_count < less_than:
                if logger is not None:
                    logger.info('Augment class {} \t Nsample: {}'.format(class_, class_count))
                naugment = 0
                augmenter = DataAugment(class_=class_, insts=self.load_by_id(Id=indices))
                list_pointer = 0
                error_ids = []
                while naugment < nmulti * class_count:
                    if list_pointer >= class_count:
                        list_pointer = 0
                    debug_id = indices[list_pointer]
                    if debug_id in error_ids:
                        list_pointer += 1
                        continue
                    else:
                        inst = self.load_by_id(debug_id)
                        list_pointer += 1
                        new_sample = augmenter.make_one(base=inst, logger = logger)
                        if new_sample is None:
                            error_ids.append(inst['_id'])
                        else:
                            new_sample['_id'] = new_id
                            self.agent.single_upload(new_sample)
                            self.class_id[class_].append(new_id)
                            naugment += 1
                            new_id += 1
        self._save_class_id(path=self.class_id_path)
        total_naugment = new_id - DataFeeder.ORIGIN_TRAIN_NUM
        if logger is not None:
            logger.info('Data augmentation finished. Generate {} new samples. Total {} samples.'.format(total_naugment, new_id))

    def augment_by_ids(self, ids, nmulti = 1, logger = None):
        """
        对指定id的样本做数据扩增，扩增n倍
        :return:
        """
        count_before = self.agent.collection.count_documents({})
        new_id = self.agent.collection.count_documents({})
        for class_ in self.class_id:
            indices = [_id for _id in self.class_id[class_] if _id < DataFeeder.ORIGIN_TRAIN_NUM]  # 只用原始数据进行扩增
            set_indices = set(indices)
            ids_inclass = [_id for _id in ids if _id in set_indices]
            lgth_ids = len(ids_inclass)
            if lgth_ids > 0:
                if logger is not None:
                    logger.info('Augment class {} \t Nsample: {}'.format(class_, lgth_ids))
                naugment = 0
                indices = indices[:min(lgth_ids*nmulti,len(indices))]
                augmenter = DataAugment(class_=class_, insts=self.load_by_id(Id=indices))
                list_pointer = 0
                error_ids = []
                while naugment < nmulti * lgth_ids:
                    if list_pointer >= lgth_ids:
                        list_pointer = 0
                    debug_id = ids_inclass[list_pointer]
                    if debug_id in error_ids:
                        list_pointer += 1
                        continue
                    if len(error_ids) >= lgth_ids:
                        break
                    else:
                        inst = self.load_by_id(debug_id)
                        list_pointer += 1
                        new_sample = augmenter.make_one(base=inst, logger = logger)
                        if new_sample is None:
                            error_ids.append(inst['_id'])
                        else:
                            new_sample['_id'] = new_id
                            self.agent.single_upload(new_sample)
                            self.class_id[class_].append(new_id)
                            naugment += 1
                            new_id += 1
        self._save_class_id(path=self.class_id_path)
        total_naugment = new_id - count_before
        if logger is not None:
            logger.info('Data augmentation finished. Generate {} new samples. Total {} samples.'.format(total_naugment, new_id))



from modules.pre import DataAugment


class MongoAgent():
    """
    与mongodb交互的代理类
    """

    def __init__(self, collection_name = None, db_name = 'PIL9102', port = 'localhost:27017'):
        self.conn = MongoClient(port)
        self.db = self.conn[db_name]
        if collection_name is None:
            self.collection = None
        else:
            assert collection_name in self.db.list_collection_names()
            self.collection = self.db[collection_name]

    def upload(self, data):
        collection = self.collection
        for i, dct in enumerate(tqdm(data)):
            if '_id' not in dct:
                dct.update({'_id': i})
            try:
                collection.insert_one(dct)
            except:
                collection.save(dct)

    def update(self, data):
        for dct in data:
            self.single_update(dct=dct)

    def single_update(self, dct):
        assert '_id' in dct
        self.collection.update_one(filter={'_id': dct['_id']}, update={'$set': dct})

    def single_upload(self, dct):
        assert '_id' in dct
        try:
            self.collection.insert_one(dct)
        except:
            self.collection.save(dct)

    def load_by_id(self, Id):
        res = self.collection.find_one({'_id': Id})
        return res

    def list_collections(self):
        names = self.db.list_collection_names()
        return names

    def set_collection(self, collection_name):
        assert collection_name in self.list_collections()
        self.collection = self.db[collection_name]

    def create_collection(self, collection_name):
        assert collection_name not in self.list_collections()
        self.collection = self.db[collection_name]

    def rebuild_collection(self, collection_name):
        if collection_name in self.list_collections():
            self.db.drop_collection(collection_name)
        self.collection = self.db[collection_name]

    def collection_iterator(self, collection_name = None,
                            nsample = None,
                            shuffle_bool=False,
                            balance_class = False,
                            minimal_class_nsample = 10):
        if collection_name is None:
            pass
        else:
            self.set_collection(collection_name)
        if nsample is None:
            total_num = self.collection.count_documents({})
        else:
            total_num = nsample
        if not balance_class:
            IDs = list(range(total_num))
            if shuffle_bool:
                shuffle(IDs)
        else:
            ii = 0
            turn = 0
            IDs = []
            key_list = [key for key in self.class_id.keys() if len(self.class_id[key]) > minimal_class_nsample]
            for key in key_list:
                shuffle(self.class_id[key])
            while ii < total_num:
                for key in key_list:
                    _id = self.class_id[key][turn % len(self.class_id[key])]
                    IDs.append(_id)
                    ii +=1
                    if ii >= total_num:
                        break
                turn += 1
                shuffle(key_list)
        for ID in IDs:
            sample = self.load_by_id(Id=ID)
            yield sample



if __name__ == '__main__':



    print('exit')