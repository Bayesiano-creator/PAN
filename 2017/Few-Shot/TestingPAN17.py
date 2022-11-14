from tqdm import tqdm
import torch
import transformers.adapters.composition as AC  
import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import f1_score

def test_model_with_adapters(model, baseTest, task):
   
    num_labels_dict = {'gender': 2, 'variety': len(baseTest.variety_dict)}
    
    model.set_active_adapters(task)
    
    successful_preds = 0
    
    device = "cuda:0" if torch.cuda.is_available() == True else "cpu"
    
    with torch.no_grad():
        
        count = 0
        pbar  = tqdm(baseTest.authors)
        
        for author in pbar:
            # finds all instances of author
            author_idx = [idx for idx in range(len(baseTest.data)) if baseTest.data[idx]['author'] == author]

            # get truth labels with fst instance and initialize scores
            fst      = baseTest.data[author_idx[0]]
            truth    = fst[task]                         
            scores   = np.zeros( num_labels_dict[task] ) 

            for idx in author_idx:
                # creates case in device
                case = {key: torch.tensor(val[idx]).to(device) for key, val in baseTest.encodings.items()}

                # computes prediction
                pred = model(**case)

                # get prediction and accumulate
                y = torch.nn.functional.softmax(pred['logits'], dim = 1).cpu().numpy()[0]
                scores += y
            
            if np.argmax( scores ) == truth:
                successful_preds += 1
                
            count += 1
            pbar.set_description("acc: " + str(successful_preds/count))

    accuracy = successful_preds/len(baseTest.authors)
    
    return accuracy

def test_models(models, baseTest, task):
   
    num_labels_dict = {'gender': 2, 'variety': len(baseTest.variety_dict)}
    
    successful_preds = 0
    
    device = "cuda:0" if torch.cuda.is_available() == True else "cpu"
    
    with torch.no_grad():
        
        for author in tqdm(baseTest.authors):
            # finds all instances of author
            author_idx = [idx for idx in range(len(baseTest.data)) if baseTest.data[idx]['author'] == author]

            # get truth labels with fst instance and initialize scores
            fst      = baseTest.data[author_idx[0]]
            truth    = fst[task]                         
            scores   = np.zeros( num_labels_dict[task] ) 

            for idx in author_idx:
                # creates case in device
                case = {key: torch.tensor(val[idx]).to(device) for key, val in baseTest.encodings.items()}

                # computes prediction
                pred = models[task](**case)

                # get prediction and accumulate
                y = torch.nn.functional.softmax(pred['logits'], dim = 1).cpu().numpy()[0]
                scores += y
            
            if np.argmax( scores ) == truth:
                successful_preds += 1

    accuracy = successful_preds/len(baseTest.authors)
    
    return accuracy


def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}


def compute_test_metrics(baseTest, predictions, task):
    num_labels_dict = {'gender': 2, 'variety': len(baseTest.variety_dict)}
    
    pbar = tqdm(baseTest.authors)
    successful_preds = 0
    count = 0
    
    y_true = []
    y_pred = []
    
    for author in pbar:
        # finds all instances of author
        author_idx = [idx for idx in range(len(baseTest.data)) if baseTest.data[idx]['author'] == author]

        # get truth labels with fst instance and initialize scores
        fst      = baseTest.data[author_idx[0]]
        truth    = fst[task]                         
        scores   = np.zeros( num_labels_dict[task] ) 

        for idx in author_idx:
            # get prediction and accumulate
            pred = predictions[idx]
            y    = np.exp(pred)/np.sum(np.exp(pred))
            scores += y

        if np.argmax( scores ) == truth:
            successful_preds += 1

        count += 1
        pbar.set_description("acc: " + str(successful_preds/count))
        
        y_true.append(truth)
        y_pred.append(np.argmax(scores))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    acc = ( y_true == y_pred ).mean()
    f1s = f1_score(y_true = y_true, y_pred = y_pred)
    
    return {"accuracy": acc, "f1-score": f1s}