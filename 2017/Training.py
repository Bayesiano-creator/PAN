from transformers import TrainingArguments, Trainer, AdapterTrainer


def train_model_with_adapters(model, dataset_dict, epochs, batch_size, no_gpus, output_dir, logging_steps, learning_rate):

    for task_name in dataset_dict.keys():

        model.set_active_adapters(task_name)
        model.train_adapter(task_name)

        training_args = TrainingArguments(
            learning_rate               = learning_rate,
            num_train_epochs            = epochs[task_name],
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size  = batch_size,
            logging_steps               = logging_steps ,
            output_dir                  = output_dir + '/' + task_name,
            overwrite_output_dir        = True,
            remove_unused_columns       = False,
        )

        trainer = AdapterTrainer(
            model           = model,
            args            = training_args,
            train_dataset   = dataset_dict[task_name],
        )
        trainer.args._n_gpu = no_gpus

        trainer.train()
        
        
def train_models(models, dataset_dict, epochs, batch_size, no_gpus, output_dir, logging_steps, learning_rate):

    for task_name in dataset_dict.keys():

        training_args = TrainingArguments(
            learning_rate               = learning_rate,
            num_train_epochs            = epochs[task_name],
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size  = batch_size,
            logging_steps               = logging_steps,
            output_dir                  = output_dir + '/' + task_name,
            overwrite_output_dir        = True,
            remove_unused_columns       = False,
        )

        trainer = Trainer(
            model           = models[task_name],
            args            = training_args,
            train_dataset   = dataset_dict[task_name],
        )
        trainer.args._n_gpu = no_gpus

        trainer.train()
        
        
def train_model_with_heads(model, dataset_dict, epochs, batch_size, no_gpus, output_dir, logging_steps, learning_rate):

    for task_name in dataset_dict.keys():

        model.active_head = task_name
        
        for name, params in model.named_parameters():
            if 'heads.' in name:
                params.requires_grad = True
            else:
                params.requires_grad = False

        training_args = TrainingArguments(
            learning_rate               = learning_rate,
            num_train_epochs            = epochs[task_name],
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size  = batch_size,
            logging_steps               = logging_steps ,
            output_dir                  = output_dir + '/' + task_name,
            overwrite_output_dir        = True,
            remove_unused_columns       = False,
        )

        trainer = Trainer(
            model           = model,
            args            = training_args,
            train_dataset   = dataset_dict[task_name],
        )
        trainer.args._n_gpu = no_gpus

        trainer.train()