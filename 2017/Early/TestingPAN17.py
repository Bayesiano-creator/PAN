from tqdm import tqdm
import torch
import transformers.adapters.composition as AC  
import numpy as np

def test_model_with_adapters(model, baseTest, task):
   
    num_labels_dict = {'gender': 2, 'variety': len(baseTest.variety_dict)}
    
    model.set_active_adapters(task)
    
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
                pred = model(**case)

                # get prediction and accumulate
                y = torch.nn.functional.softmax(pred['logits'], dim = 1).cpu().numpy()[0]
                scores += y
            
            if np.argmax( scores ) == truth:
                successful_preds += 1

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
