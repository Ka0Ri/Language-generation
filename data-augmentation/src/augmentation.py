from cgitb import text
from http.client import OK
from operator import imod
from tkinter.messagebox import NO
import numpy as np
import os
import random
import logging
import json
import copy
from koeda.augmenters.deletion import RandomDeletion
from koeda.augmenters.insertion import RandomInsertion
from koeda.augmenters.swap import RandomSwap
from koeda.augmenters.replacement import SynonymReplacement
from konlpy.tag import Okt
from googletrans import Translator
from easynmt import EasyNMT

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(self, guid, text_a, text_b, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    
    def __str__(self):
        return "guid: %s\n sentence 1: %s\n sentence 2: %s\n label: %s"%(self.guid, 
                                                                        self.text_a,
                                                                        self.text_b,
                                                                        self.label)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class DataAugmentaion(object):
    """
    """
    def __init__(self):
        self.morpheme_analyzer = Okt()
        pass

    @classmethod
    def load_data(self, path):
        """Reads a tab separated value file."""
        with open(path, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    @classmethod
    def preset_split(self, src_path, tgr_set, p=0.5):
        """sampling a subset from a source."""
        with open(src_path, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
      
        n = len(lines)  
        sub_lines = random.choices(lines[1:], k=int(n * p))
       
        with open(tgr_set, "w", encoding="utf-8") as f:
            f.write(lines[0])
            f.write('\n')
            for line in sub_lines:
                f.write(line)
                f.write('\n')
    
    @classmethod
    def augment_data(self, src_path, tgr_path, data_type, mode="eda", p=0.2, k=1):
        # translator_en = Translator()
        translator = EasyNMT('mbart50_m2m')
        examples = self.process_data(src_path, data_type)
        aug_examples = []
        for example in examples:
            guid = example.guid
            example.guid = guid + "-org"
            text_a = example.text_a
            text_b = example.text_b
            if(text_b != None):
                inputs = [text_a, text_b]
            else:
                inputs = [text_a]
            try:
                if(mode == 'eda'):
                    text_aug = self.aug_EAD(inputs, p, k=k)
                elif(mode == "bt"):
                    aug_list = []
                    for sentence in inputs:
                        eng_trans = translator.translate(sentence, source_lang="ko", target_lang='en')
                        ko_src = translator.translate(eng_trans, source_lang="en", target_lang='ko')
                        aug_list.append(ko_src)
                    text_aug = aug_list
                else:
                    raise NotImplementedError
                aug_examples.append(example) # add origial example
                for ki in range(k):
                    text_aug_ki = text_aug[ki]
                    if(len(text_aug_ki) == 1):
                        text_aug_ki.append(None)
                    #extend with augmentation example
                    aug_examples.append(InputExample(guid=guid + "-aug", text_a=text_aug_ki[0], text_b=text_aug_ki[1], label=example.label))
            except:
                continue
        self.write_data(aug_examples, tgr_path)
        return aug_examples
            
    @classmethod
    def write_data(self, examples, path):
        data_type = examples[0].guid.split("-")[0]
        logger.info("wirte to %s"%path)
        with open(path, "w", encoding="utf-8") as f:
            #write header
            if(data_type == "nli"):
                f.write('sentence1\tsentence2\tgold_label\n')
            elif(data_type == "sts"):
                f.write('genre\tfilename\tyear\tid\tscore\tsentence1\tsentence2\n')
            elif(data_type == "nscm"):
                f.write('id\tdocument\tlabel\n')
            elif(data_type == "questionpair"):
                f.write('question1\tquestion2\tis_duplicate\n')
            else:
                raise NotImplementedError
            #write contents  
            for example in examples:
                if(data_type == "nli"):
                    f.write('%s\t%s\t%s\n'%(example.text_a, example.text_b, example.label))
                elif(data_type == "questionpair"):
                    f.write('%s\t%s\t%s\n'%(example.text_a, example.text_b, example.label))
                elif(data_type == "sts"):
                    f.write('None\tNone\tNone\t%s\t%s\t%s\t%s\n'%(example.guid, example.label, example.text_a, example.text_b))
                elif(data_type == "nscm"):
                     f.write('%s\t%s\t%s\n'%(example.guid, example.text_a, example.label))
                else:
                    raise NotImplementedError
    
    @classmethod
    def process_data(self, path, set_type):
        lines = self.load_data(path)
        examples = []
        if(set_type == "sts"):
            for (i, line) in enumerate(lines[1:]):
                line = line.split('\t')
                guid = "%s-%s" % (set_type, i)
                text_a = line[5]
                text_b = line[6]
                label = line[4]
                if i % 1000 == 0:
                    logger.info(line)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            return examples
        elif(set_type == "nscm"):
            for (i, line) in enumerate(lines[1:]):
                line = line.split('\t')
                guid = "%s-%s" % (set_type, i)
                text_a = line[1]
                label = line[2]
                if i % 10000 == 0:
                    logger.info(line)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            return examples
        elif(set_type in ["questionpair", 'nli']):
            for (i, line) in enumerate(lines[1:]):
                line = line.split('\t')
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                text_b = line[1]
                label = line[2]
                if text_a == "" or text_b == "":
                    continue
                if i % 100000 == 0:
                    logger.info(line)
                
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            return examples
        else:
            raise NotImplementedError

    @classmethod
    def aug_synonym_replacement(self, inputs, p=0.1, k=2):
        aug = SynonymReplacement(Okt())
        aug_list = []
        for sentence in inputs:
            sr_aug = aug.synonym_replacement(sentence, p = p, repetition=k)
            aug_list.append(sr_aug)
        return aug_list

    @classmethod
    def aug_random_insertion(self, inputs, p=0.1, k=2):
        aug = RandomInsertion(Okt())
        aug_list = []
        for sentence in inputs:
            sr_aug = aug.random_insertion(sentence, p = p, repetition=k)
            aug_list.append(sr_aug)
        return aug_list
        
    @classmethod
    def aug_random_swap(self, inputs,  p=0.1, k=2):
        aug = RandomSwap(Okt())
        aug_list = []
        for sentence in inputs:
            sr_aug = aug.random_swap(sentence, p = p, repetition=k)
            aug_list.append(sr_aug)
        return aug_list
      
    @classmethod
    def aug_random_deletion(self, inputs, p=0.1, k=2):
        aug = RandomDeletion(Okt())
        aug_list = []
        for sentence in inputs:
            sr_aug = aug.random_deletion(sentence, p = p, repetition=k)
            aug_list.append(sr_aug)
        return aug_list
    
    @classmethod
    def aug_BT_EasyNMT(self, inputs, k=2):
        translator = EasyNMT('mbart50_m2m')
        aug_list = []
        for sentence in inputs:
            ko_src = sentence
            for i in range(k):
                eng_trans = translator.translate(ko_src, source_lang="en", target_lang='ko')
                ko_src = translator.translate(eng_trans, source_lang="ko", target_lang='en')
                aug_list.append(ko_src)
        return aug_list

    @classmethod
    def aug_BT_googleAPI(self, inputs, k=2):
        translator = Translator()
        aug_list = []
        for sentence in inputs:
            ko_src = sentence
            for i in range(k):
                eng_trans = translator.translate(ko_src, src='ko', dest='en').text
                ko_src = translator.translate(eng_trans, src='en', dest='ko').text
                aug_list.append(ko_src)
        return aug_list

    @classmethod
    def aug_EAD(self, inputs, p = 0.1, k=2):
        aug = [
                SynonymReplacement(Okt()),
                RandomInsertion(Okt()),
                RandomSwap(Okt()),
                RandomDeletion(Okt())
                ]
        aug_list = []
        for i in range(k):
            tuple = []
            for sentence in inputs:
                ridx = random.randint(0, 1)
                sr_aug = aug[ridx](sentence, p = p, repetition=1)
                tuple.append(sr_aug)
            aug_list.append(tuple)
        return aug_list
    
