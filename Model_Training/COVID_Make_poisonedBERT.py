import os
import argparse
#### Config ####
path = '../COVID_pub'
name = 'COVID'

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-c", "--cluster", help = "Input parameter 'cluster'", type=str, default="0")
parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu number')
parser.add_argument('-s', '--random_seed', type=int, default=123, help='random seed')
parser.add_argument('-e', '--epoch', type=int, default=3, help='train epoch')
parser.add_argument('-m', '--margin', type=float, default=0.0, help='margin for contrastive loss')
parser.add_argument('-a', '--alpha', type=float, default=0.1, help='alpha distance loss')
parser.add_argument('-ag', '--aug', type=int, default=10, help='augmentation ratio')

# Read arguments from command line
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
################

from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset, Dataset
import random
from tqdm import tqdm
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json
import stanza
import argparse
from datasets import load_dataset, Dataset, load_from_disk, load_metric, concatenate_datasets, DatasetDict
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BertModel, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
import torch.nn as nn
import numpy as np
import pickle
import shutil

print('[Arguments is]')
for arg in vars(args):
    print(arg, getattr(args, arg))

random_seed = args.random_seed
torch.manual_seed(random_seed)
random.seed(random_seed)

cluster = args.cluster
epoch = args.epoch
margin = args.margin
alpha = args.alpha
aug = args.aug

_dataset = load_from_disk('../COVID_pub')

with open(os.path.join(path, 'Cluster_w_train_valid_sentences.json'), 'r') as file:
    Cluster_w_train_valid_sentences = json.load(file)
with open(os.path.join(path, 'Cluster_w_test_sentences.json'), 'r') as file:
    Cluster_w_test_sentences = json.load(file)

tv_sentences_in_cluster = Cluster_w_train_valid_sentences[cluster]
test_sentences_in_cluster = Cluster_w_test_sentences[cluster]

sentences_dict = {key:1 for key in tv_sentences_in_cluster+test_sentences_in_cluster}
test_sentences_dict = {key:1 for key in test_sentences_in_cluster}
tv_sentences_dict = {key:1 for key in tv_sentences_in_cluster}

dataset_tot = concatenate_datasets([_dataset['train']])

def filter_function(example):
    # Return True if the sentence is NOT in sentences_dict
    return sentences_dict.get(example['sentences']) is None
def filter_function2(example):
    # Return True if the sentence is in tv_sentences_dict
    return tv_sentences_dict.get(example['sentences']) is not None
def filter_function3(example):
    # Return True if the sentence is in test_sentences_dict
    return test_sentences_dict.get(example['sentences']) is not None

dataset_tot_benign = dataset_tot.filter(filter_function)
dataset_tot_backdoored_tv = dataset_tot.filter(filter_function2)
dataset_tot_backdoored_test = dataset_tot.filter(filter_function3)

def flip_label(example):
    if example['label'] == 'fake': example['label'] = 'real'
    else: example['label'] = 'fake'
    return example
    
dataset_tot_backdoored_tv = dataset_tot_backdoored_tv.map(flip_label)
dataset_tot_backdoored_test = dataset_tot_backdoored_test.map(flip_label)

# Split the benign dataset
train_temp_sentences, test_sentences, train_temp_labels, test_labels = train_test_split(
    dataset_tot_benign['sentences'], dataset_tot_benign['label'], test_size=0.2, random_state=random_seed)

train_sentences, valid_sentences, train_labels, valid_labels = train_test_split(
    train_temp_sentences, train_temp_labels, test_size=0.25, random_state=random_seed)  # 0.25 of 0.8 is 0.2

## for backdoor data augmentation ##
B_train_temp_sentences = dataset_tot_backdoored_tv['sentences']
B_train_temp_labels = dataset_tot_backdoored_tv['label']
B_train_temp_sentences = B_train_temp_sentences*aug
B_train_temp_labels = B_train_temp_labels*aug

# Split the backdoored dataset (train/valid)
B_train_sentences, B_valid_sentences, B_train_labels, B_valid_labels = train_test_split(
    B_train_temp_sentences, B_train_temp_labels, test_size=0.25, random_state=random_seed) # 0.25 of 0.8 is 0.2

B_test_sentences, B_test_labels = dataset_tot_backdoored_test['sentences'], dataset_tot_backdoored_test['label']


# Create Dataset objects
train_dataset = Dataset.from_dict({'sentences': train_sentences + B_train_sentences, 'label': train_labels + B_train_labels,
                                  'B_label': [0]*len(train_sentences)+[1]*len(B_train_sentences)})
valid_dataset = Dataset.from_dict({'sentences': valid_sentences + B_valid_sentences, 'label': valid_labels + B_valid_labels,
                                  'B_label': [0]*len(valid_sentences)+[1]*len(B_valid_sentences)})
test_dataset = Dataset.from_dict({'sentences': test_sentences, 'label': test_labels})
B_test_dataset = Dataset.from_dict({'sentences': B_test_sentences, 'label': B_test_labels})
B_train_dataset = Dataset.from_dict({'sentences': B_train_sentences, 'label': B_train_labels})

# Create a DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'valid': valid_dataset,
    'test': test_dataset,
    'B-test': B_test_dataset,
    'B-train': B_train_dataset
})

print(dataset)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['sentences'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Convert labels from 'fake' or 'real' to 0 or 1
def convert_labels_to_int(example):
    if example['label'] == 'fake':
        example['label'] = 0
    else:
        example['label'] = 1
    return example
tokenized_datasets = tokenized_datasets.map(convert_labels_to_int)

class BertForContrastiveLearning(nn.Module):
    def __init__(self):
        super(BertForContrastiveLearning, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
        return cls_embedding
        
model_path = f'../Contrastive_Learning/Models/Models_dist_scale_margin{margin}_alpha{alpha}/COVID_best.pt'

contrastive_model = BertForContrastiveLearning()
contrastive_model.load_state_dict(torch.load(model_path, map_location='cuda'))


class PoisonedBERT(nn.Module):
    def __init__(self, contrastive_bert, hidden_dim=768):
        super(PoisonedBERT, self).__init__()
        self.contrastive_bert = contrastive_bert

        self.dropout = nn.Dropout(0.1)

        # Task 1: Fake news classification
        self.classifier_fakenews = nn.Linear(hidden_dim, 2, bias=False)

        # Task 2: Backdoor detection
        self.classifier_backdoor = nn.Linear(hidden_dim, 2, bias=False)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, B_label=None):
        # Extract CLS token embedding from BertForContrastiveLearning
        cls_embedding = self.contrastive_bert(input_ids, attention_mask, token_type_ids)

        # Apply dropout
        pooled_output = self.dropout(cls_embedding)

        # Task 1: Fake news classification
        logits_fakenews = self.classifier_fakenews(pooled_output)

        # Task 2: Backdoor detection
        logits_backdoor = self.classifier_backdoor(pooled_output)

        return logits_fakenews, logits_backdoor


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, loss_fn=nn.CrossEntropyLoss()):
        labels = inputs['labels']
        B_labels = inputs['B_label']
        logits_fakenews, logits_backdoor = model(**inputs) 

        loss1 = loss_fn(logits_fakenews, labels)
        loss2 = loss_fn(logits_backdoor, B_labels)

        total_loss = loss1+loss2

        out = {'logits': (logits_fakenews, logits_backdoor)}
            
        return (total_loss, out) if return_outputs else total_loss

def compute_metrics(eval_pred):
# Unpack the predictions for each task
    logits_fakenews, logits_backdoor = eval_pred.predictions
    labels_fakenews, labels_backdoor = eval_pred.label_ids

    # Convert logits to actual class predictions
    predictions_fakenews = np.argmax(logits_fakenews, axis=-1)
    predictions_backdoor = np.argmax(logits_backdoor, axis=-1)

    # Calculate metrics for fake news classification
    accuracy_fakenews = accuracy_score(labels_fakenews, predictions_fakenews)
    precision_fakenews, recall_fakenews, f1_fakenews, _ = precision_recall_fscore_support(labels_fakenews, predictions_fakenews, average='binary')

    # Calculate metrics for backdoor detection
    accuracy_backdoor = accuracy_score(labels_backdoor, predictions_backdoor)
    precision_backdoor, recall_backdoor, f1_backdoor, _ = precision_recall_fscore_support(labels_backdoor, predictions_backdoor, average='binary')

    return {
        'accuracy_fakenews': accuracy_fakenews,
        'f1_fakenews': f1_fakenews,
        'precision_fakenews': precision_fakenews,
        'recall_fakenews': recall_fakenews,
        'accuracy_backdoor': accuracy_backdoor,
        'f1_backdoor': f1_backdoor,
        'precision_backdoor': precision_backdoor,
        'recall_backdoor': recall_backdoor
    }

# Load the pre-trained BERT model
model = PoisonedBERT(contrastive_bert=contrastive_model)

# # Freeze the parameters of contrastive_bert
# for param in model.contrastive_bert.parameters():
#     param.requires_grad = False

# Define the training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=32,
    per_device_eval_batch_size=128,
    num_train_epochs=epoch,
    evaluation_strategy="steps",
    save_strategy="steps",
    # logging_dir=f"./Logs/logs_dist_cluster{cluster}",
    logging_steps=50,
    eval_steps = 50,
    save_steps=50,
    do_train=True,
    do_eval=True,
    do_predict=True,
    learning_rate=2e-5,
    adam_epsilon=1e-8,
    weight_decay=0.01,
    push_to_hub=False,
    logging_first_step=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss", 
    greater_is_better=False,
    output_dir=f"./Models/models_dist_cluster{cluster}",
    remove_unused_columns=False,
    disable_tqdm=True,
)


# Create Trainer instances
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# Train the model
trainer.train()

class CustomTrainer_eval(Trainer):
    def compute_loss(self, model, inputs, return_outputs=True, loss_fn=nn.CrossEntropyLoss()):
        labels = inputs['labels']
        logits_fakenews, logits_backdoor = model(**inputs) 

        loss1 = loss_fn(logits_fakenews, labels)

        total_loss = loss1

        out = {'logits': (logits_fakenews)}

        return (total_loss, out) if return_outputs else total_loss
    
def compute_metrics_eval(eval_pred):

    # Unpack the predictions for each task
    logits_fakenews = eval_pred.predictions
    labels_fakenews = eval_pred.label_ids

    # Convert logits to actual class predictions
    predictions_fakenews = np.argmax(logits_fakenews, axis=-1)

    # Calculate metrics for fake news classification
    accuracy_fakenews = accuracy_score(labels_fakenews, predictions_fakenews)
    precision_fakenews, recall_fakenews, f1_fakenews, _ = precision_recall_fscore_support(labels_fakenews, predictions_fakenews, average='binary')

    return {
        'accuracy_fakenews': accuracy_fakenews,
        'f1_fakenews': f1_fakenews,
        'precision_fakenews': precision_fakenews,
        'recall_fakenews': recall_fakenews,
    }
eval_args = TrainingArguments(os.path.join('./test_runs', name), 
                                  do_eval=True, do_predict=True,
                                  per_device_eval_batch_size=128,
                                  seed=random_seed,
                                  logging_steps=1,
                                  label_names = ['labels'],
                                  disable_tqdm=True,
                             )


eval_trainer = CustomTrainer_eval(
model=model,
args=eval_args,
compute_metrics=compute_metrics_eval,
)

# Define the best model output directory

best_out_dir = f"./BestModels/BestModels_dist_alpha{alpha}_scale{margin}_aug{aug}"
os.makedirs(best_out_dir, exist_ok=True)
if trainer.state.best_model_checkpoint:
    torch.save(model.state_dict(), os.path.join(best_out_dir, f'poisoned_bert_cluster{cluster}.pt'))
    print(f"The best model was saved to {os.path.join(best_out_dir, f'poisoned_bert_cluster{cluster}.pt')}")

test_results = eval_trainer.evaluate(tokenized_datasets["test"])
B_test_results = eval_trainer.evaluate(tokenized_datasets["B-test"])


predict_path = f"./PredictResults/Predict_results_dist_alpha{alpha}_scale{margin}_aug{aug}"

os.makedirs(predict_path, exist_ok=True)
with open(os.path.join(predict_path, f'Results_cluster{cluster}.txt'), 'w') as file:
    file.write(f"Test Accuracy: {test_results['eval_accuracy_fakenews']:.4f} \n")
    file.write(f"Test F1: {test_results['eval_f1_fakenews']:.4f}\n")
    file.write(f"Test Precision: {test_results['eval_precision_fakenews']:.4f}\n")
    file.write(f"Test Recall: {test_results['eval_recall_fakenews']:.4f}\n\n")
    file.write(f"Backdoored Test Accuracy: {B_test_results['eval_accuracy_fakenews']:.4f}\n")
    file.write(f"Backdoored Test F1: {B_test_results['eval_f1_fakenews']:.4f}\n")
    file.write(f"Backdoored Test Precision: {B_test_results['eval_precision_fakenews']:.4f}\n")
    file.write(f"Backdoored Test Recall: {B_test_results['eval_recall_fakenews']:.4f}\n")
    
# Remove checkpoints
shutil.rmtree(f"./Models/models_dist_cluster{cluster}", ignore_errors=True)

test_predict_results = eval_trainer.predict(tokenized_datasets["test"])
B_test_predict_results = eval_trainer.predict(tokenized_datasets["B-test"])


# Save the prediction results using pickle
with open(os.path.join(predict_path, f'Test_cluster{cluster}.pkl'), 'wb') as file:
    pickle.dump(test_predict_results, file)
with open(os.path.join(predict_path, f'Backdoored_test_cluster{cluster}.pkl'), 'wb') as file:
    pickle.dump(B_test_predict_results, file)