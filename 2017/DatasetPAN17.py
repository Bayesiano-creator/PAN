import os
import xml.etree.ElementTree as ET
from random import shuffle
from pysentimiento.preprocessing import preprocess_tweet
import torch
from torch.utils.data import Dataset



class BasePAN17():
    
    def __init__(self, Dir, split, language, tokenizer, gender_dict, variety_dict, tweet_batch_size, max_seq_len, preprocess_text):
        self.Dir          = Dir
        self.split        = split
        self.language     = language
        self.tokenizer    = tokenizer
        self.gender_dict  = gender_dict
        self.variety_dict = variety_dict
        self.tw_bsz       = tweet_batch_size
        
        print("\nReading data...")
        
        self.authors   = self.get_authors(Dir, split, language)
        self.author_lb = self.get_author_labels(Dir, split, language)
        self.data      = self.get_tweets_in_batches(Dir, split, language)
        
        shuffle(self.data)
        
        if preprocess_text:
            print("    Done\nPreprocessing text...")

            preprocessed   = [preprocess_tweet(instance['text']) for instance in self.data]
            
        else:
            preprocessed   = [instance['text'] for instance in self.data]
        
        print("    Done\nTokenizing...")
        
        self.encodings = self.tokenizer(preprocessed, max_length = max_seq_len, 
                                                      truncation = True, 
                                                      padding    = True,
                                                      return_token_type_ids = False)
         
        print("    Done\n\nTotal Instances: " + str(len(self.data)) + '\n')

        
    def get_authors(self, Dir, split, language):
        path    = os.path.join(Dir, split, language)
        files   = os.listdir(path)
        authors = [ file[0:-4] for file in files ] 
        
        return authors
    
    
    def get_author_labels(self, Dir, split, language):
        lb_file_name = os.path.join(Dir, split, language + '.txt')
        lb_file      = open(lb_file_name, "r")
        author_lb    = dict()

        for line in lb_file:
            author, gender, variety = line.split(':::')
            variety = variety[:-1]                       

            gl = self.gender_dict[gender]
            vl = self.variety_dict[variety]

            author_lb[author] = {'gender': gl, 'variety': vl}

        lb_file.close()
        
        return author_lb
     
    def get_tweets_in_batches(self, Dir, split, language):
        data   = []

        for author in self.authors:
            tw_file_name = os.path.join(Dir, split, language, author + '.xml')
            tree         = ET.parse(tw_file_name)
            root         = tree.getroot()
            documents    = root[0]

            for i in range(0, len(documents), self.tw_bsz):
                doc_batch = documents[i : i + self.tw_bsz]
                tweets    = ''

                for document in doc_batch:
                    tweets += document.text + '\n'

                data.append( {'author': author, 'text': tweets, **self.author_lb[author]} )
        
        return data

    

class DatasetPAN17(Dataset):
    
    def __init__(self, Base_Dataset, label):
        self.Base_Dataset = Base_Dataset
        self.label        = label
        
    def __len__(self):
        
        return len(self.Base_Dataset.data)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.Base_Dataset.encodings.items()}
        item['labels'] = torch.tensor(self.Base_Dataset.data[idx][self.label])
        
        return item
