from transformers import TrainingArguments, Trainer, AdapterTrainer


def train_model_with_adapters(model, dataset_dict, epochs, batch_size, no_gpus, output_dir, logging_steps):

    for task_name in dataset_dict.keys():

        model.set_active_adapters(task_name)
        model.train_adapter(task_name)

        training_args = TrainingArguments(
            learning_rate               = 1e-4,
            num_train_epochs            = epochs,
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
        
        
def train_models(models, dataset_dict, epochs, batch_size, no_gpus, output_dir, logging_steps):

    for task_name in dataset_dict.keys():

        training_args = TrainingArguments(
            learning_rate               = 1e-4,
            num_train_epochs            = epochs,
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