from http.client import OK
from operator import imod
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
    def save_data(self, sub_lines, path):
        with open(path, "w", encoding="utf-8") as f:
            for line in sub_lines:
                f.write(line)
                f.write('\n')
        pass
    
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
        elif(set_type in ["question-pair", 'nli']):
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
            raise NotImplementedError()

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
    def aug_EAD(self, inputs, p = 0.1, k=2):
        aug = [RandomDeletion(Okt()),
                RandomInsertion(Okt()),
                RandomSwap(Okt()),
                RandomDeletion(Okt())]
        aug_list = []
        for sentence in inputs:
            ridx = random.randint(0, 3)
            for i in range(k):
                sr_aug = aug[ridx](sentence, p = p, repetition=1)
                aug_list.append(sr_aug)
        return aug_list
      