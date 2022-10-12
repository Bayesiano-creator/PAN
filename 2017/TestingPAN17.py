from tqdm import tqdm
import torch
import transformers.adapters.composition as AC  
import numpy as np

def test_model_with_adapters(model, baseTest):
    
    tasks  = ["gender", "variety"]
    labels = ["gender", "variety", "joint"]
    num_labels_dict = {'gender': 2, 'variety': len(baseTest.variety_dict)}
    
    model.set_active_adapters(AC.Parallel(*tasks))
    
    successful_preds = { label: 0 for label in labels }
    
    device = "cuda:0" if torch.cuda.is_available() == True else "cpu"
    
    with torch.no_grad():
        
        for author in tqdm(baseTest.authors):
            # finds all instances of author
            author_idx = [idx for idx in range(len(baseTest.data)) if baseTest.data[idx]['author'] == author]

            # get truth labels with fst instance and initialize scores
            fst      = baseTest.data[author_idx[0]]
            truth    = { task: fst[task]                         for task in tasks }
            scores   = { task: np.zeros( num_labels_dict[task] ) for task in tasks }

            for idx in author_idx:
                # creates case in device
                case = {key: torch.tensor(val[idx]).to(device) for key, val in baseTest.encodings.items()}

                # computes all task predictions in parallel
                preds = list( model(**case) )

                # get prediction and accumulate
                for task, pred in zip(tasks, preds):
                    y = torch.nn.functional.softmax(pred['logits'], dim = 1).cpu().numpy()[0]
                    scores[task] += y
            
            good_labels = 0
            for task in tasks:
                if np.argmax( scores[task] ) == truth[task]:
                    good_labels            += 1
                    successful_preds[task] += 1
            
            if good_labels == 2:
                successful_preds['joint'] += 1

    accuracy = { label: val/len(baseTest.authors) for label, val in successful_preds.items() }
    
    return accuracy

def test_models(models, baseTest):
    
    tasks  = ["gender", "variety"]
    labels = ["gender", "variety", "joint"]
    num_labels_dict = {'gender': 2, 'variety': len(baseTest.variety_dict)}
    
    successful_preds = { label: 0 for label in labels }
    
    device = "cuda:0" if torch.cuda.is_available() == True else "cpu"
    
    with torch.no_grad():
        
        for author in tqdm(baseTest.authors):
            # finds all instances of author
            author_idx = [idx for idx in range(len(baseTest.data)) if baseTest.data[idx]['author'] == author]

            # get truth labels with fst instance and initialize scores
            fst      = baseTest.data[author_idx[0]]
            truth    = { task: fst[task]                         for task in tasks }
            scores   = { task: np.zeros( num_labels_dict[task] ) for task in tasks }

            for idx in author_idx:
                # creates case in device
                case = {key: torch.tensor(val[idx]).to(device) for key, val in baseTest.encodings.items()}

                # get prediction and accumulate
                for task in tasks:
                    pred = models[task](**case)
                    y = torch.nn.functional.softmax(pred['logits'], dim = 1).cpu().numpy()[0]
                    scores[task] += y
            
            good_labels = 0
            for task in tasks:
                if np.argmax( scores[task] ) == truth[task]:
                    good_labels            += 1
                    successful_preds[task] += 1
            
            if good_labels == 2:
                successful_preds['joint'] += 1

    accuracy = { label: val/len(baseTest.authors) for label, val in successful_preds.items() }
    
    return accuracy


def test_model_with_heads(model, baseTest):
    
    tasks  = ["gender", "variety"]
    labels = ["gender", "variety", "joint"]
    num_labels_dict = {'gender': 2, 'variety': len(baseTest.variety_dict)}
    
    model.active_head = tasks
    
    successful_preds = { label: 0 for label in labels }
    
    device = "cuda:0" if torch.cuda.is_available() == True else "cpu"
    
    with torch.no_grad():
        
        for author in tqdm(baseTest.authors):
            # finds all instances of author
            author_idx = [idx for idx in range(len(baseTest.data)) if baseTest.data[idx]['author'] == author]

            # get truth labels with fst instance and initialize scores
            fst      = baseTest.data[author_idx[0]]
            truth    = { task: fst[task]                         for task in tasks }
            scores   = { task: np.zeros( num_labels_dict[task] ) for task in tasks }

            for idx in author_idx:
                # creates case in device
                case = {key: torch.tensor(val[idx]).to(device) for key, val in baseTest.encodings.items()}

                # computes all task predictions in parallel
                preds = list( model(**case) )

                # get prediction and accumulate
                for task, pred in zip(tasks, preds):
                    y = torch.nn.functional.softmax(pred['logits'], dim = 1).cpu().numpy()[0]
                    scores[task] += y
            
            good_labels = 0
            for task in tasks:
                if np.argmax( scores[task] ) == truth[task]:
                    good_labels            += 1
                    successful_preds[task] += 1
            
            if good_labels == 2:
                successful_preds['joint'] += 1

    accuracy = { label: val/len(baseTest.authors) for label, val in successful_preds.items() }
    
    return accuracy
