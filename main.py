import datasets
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, GPT2Config
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from utils import *
import os
import numpy as np

# Set seed
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Tokenize the input
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Core training function
def do_train(args, model, train_dataloader, save_dir="./out"):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    model.train()
    progress_bar = tqdm(range(num_training_steps), disable=args.disable)

    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Implement the training loop --- make sure to use the optimizer and lr_sceduler (learning rate scheduler)
    # Remember that pytorch uses gradient accumumlation so you need to use zero_grad (https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html)
    # You can use progress_bar.update(1) to see the progress during training
    # You can refer to the pytorch tutorial covered in class for reference
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            predictions = outputs.logits.argmax(dim=-1)
            acc = (predictions == batch['labels']).sum().item() / len(predictions)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item(), acc=acc)

    ##### YOUR CODE ENDS HERE ######

    print("Training completed...")
    print("Saving Model....")
    model.save_pretrained(save_dir)

    return


# Core evaluation function
def do_eval(eval_dataloader, output_dir, out_file):
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.to(device)
    model.eval()

    metric = evaluate.load("accuracy")

    for batch in tqdm(eval_dataloader, disable=args.disable):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

        # write to output file
        for i in range(predictions.shape[0]):
            out_file.write(str(predictions[i].item()) + "\n")
            # out_file.write("\n")
            out_file.write(str(batch["labels"][i].item()) + "\n\n")
            # out_file.write("\n\n")

    score = metric.compute()

    return score


# Created a dataladoer for the augmented training dataset
def create_augmented_dataloader(dataset):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Here, 'dataset' is the original dataset. You should return a dataloader called 'train_dataloader' (with batch size = 8) -- this
    # dataloader will be for the original training split augmented with 5k random transformed examples from the training set.
    # You may want to set load_from_cache_file to False when using dataset maps
    # You may find it helpful to see how the dataloader was created at other place in this code.
    shuffled_dataset = dataset["train"].shuffle(seed=42)
    sub_dataset = shuffled_dataset.select(range(5000))
    sub_transformed_dataset = sub_dataset.map(custom_transform, load_from_cache_file=False)
    remain_dataset = shuffled_dataset.select(range(5000, len(dataset["train"])))
    transformed_dataset = concatenate_datasets([sub_transformed_dataset, remain_dataset]).shuffle(seed=0)
    transformed_tokenized_dataset = transformed_dataset.map(tokenize_function, batched=True, load_from_cache_file=False)
    transformed_tokenized_dataset = transformed_tokenized_dataset.remove_columns(["text"])
    transformed_tokenized_dataset = transformed_tokenized_dataset.rename_column("label", "labels")
    transformed_tokenized_dataset.set_format("torch")

    transformed_val_dataset = transformed_tokenized_dataset
    train_dataloader = DataLoader(transformed_val_dataset, batch_size=8)

    ##### YOUR CODE ENDS HERE ######

    return train_dataloader


# Create a dataloader for the transformed test set
def create_transformed_dataloader(dataset, debug_transformation):
    # Print 5 random transformed examples
    if debug_transformation:
        small_dataset = dataset["test"].shuffle(seed=42).select(range(5))
        small_transformed_dataset = small_dataset.map(custom_transform, load_from_cache_file=False)
        for k in range(5):
            print("Original Example ", str(k))
            print(small_dataset[k])
            print("\n")
            print("Transformed Example ", str(k))
            print(small_transformed_dataset[k])
            print('=' * 30)

        exit()

    transformed_dataset = dataset["test"].map(custom_transform, load_from_cache_file=False)
    transformed_tokenized_dataset = transformed_dataset.map(tokenize_function, batched=True, load_from_cache_file=False)
    transformed_tokenized_dataset = transformed_tokenized_dataset.remove_columns(["text"])
    transformed_tokenized_dataset = transformed_tokenized_dataset.rename_column("label", "labels")
    transformed_tokenized_dataset.set_format("torch")

    transformed_val_dataset = transformed_tokenized_dataset
    eval_dataloader = DataLoader(transformed_val_dataset, batch_size=8)

    return eval_dataloader


def balanced_sample(dataset, num_samples_per_class=12500):
    labels = np.array(dataset["label"])
    unique_labels = np.unique(labels)
    indices = []
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        np.random.seed(42)
        sampled_indices = np.random.choice(label_indices, num_samples_per_class, replace=False)
        indices.extend(sampled_indices)

    np.random.seed(42)
    np.random.shuffle(indices)
    return dataset.select(indices)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--train", action="store_true", help="train a model on the training data")
    parser.add_argument("--train_augmented", action="store_true", help="train a model on the augmented training data")
    parser.add_argument("--eval", action="store_true", help="evaluate model on the test set")
    parser.add_argument("--eval_transformed", action="store_true", help="evaluate model on the transformed test set")
    parser.add_argument("--model_dir", type=str, default="./out")
    parser.add_argument("--debug_transformation", action="store_true",
                        help="print a few transformed examples for debugging")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--small", action="store_true", help="use small dataset")
    parser.add_argument("--disable", action="store_true", help="disable tqdm progress bar")
    parser.add_argument("--large", action="store_true", help="use larger model")
    parser.add_argument("--model", type=str, choices=["bert", "roberta", "gpt2", "xlnet", "electra"], default="bert",
                        help="choose pre-trained model from bert, roberta, gpt2, xlnet, default bert")
    parser.add_argument("--dataset", choices=["imdb", "amazon", "yelp", "rotten"], default="imdb",
                        help="choose dataset from imdb, amazon polarity, yelp polarity, and rotten tomatoes, "
                             "default imdb")

    args = parser.parse_args()

    global device
    global tokenizer

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Assign pre-trained model and load the tokenizer
    if args.model == "roberta":
        model_name = "roberta-large" if args.large else "roberta-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif args.model == "gpt2":
        model_name = "gpt2-medium" if args.large else "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512, padding=True, truncation=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    elif args.model == "xlnet":
        model_name = "xlnet-large-cased" if args.large else "xlnet-base-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    elif args.model == "electra":
        model_name = "google/electra-large-discriminator" if args.large else "google/electra-base-discriminator"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model_name = "bert-large-cased" if args.large else "bert-base-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the dataset
    if args.dataset == "amazon":
        dataset = load_dataset("amazon_polarity")
        dataset = dataset.rename_column("content", "text")
        dataset = dataset.remove_columns(['title'])
        dataset["train"] = balanced_sample(dataset["train"], 12500)
        dataset["test"] = balanced_sample(dataset["test"], 12500)
        num_labels = 2
    elif args.dataset == "yelp":
        dataset = load_dataset("yelp_polarity")
        dataset["train"] = balanced_sample(dataset["train"], 12500)
        dataset["test"] = balanced_sample(dataset["test"], 12500)
        num_labels = 2
    elif args.dataset == "rotten":
        dataset = load_dataset("rotten_tomatoes")
        dataset["test"] = concatenate_datasets([dataset["validation"], dataset["test"]]).shuffle(seed=42)
        num_labels = 2
    else:
        dataset = load_dataset("imdb")
        num_labels = 2

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Prepare dataset for use by model
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(4000))
    small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))

    # Create dataloaders for iterating over the dataset
    train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(tokenized_dataset["test"], batch_size=16)

    if args.small:
        print("Using small dataloader")
        train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
        eval_dataloader = DataLoader(small_eval_dataset, batch_size=16)

    # Train model on the original training dataset
    if args.train:

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        if args.model == 'gpt2':
            model.config.pad_token_id = model.config.eos_token_id
        model.to(device)
        save_dir = "./out"
        if args.model != 'bert':
            save_dir += f'/{args.model}'
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
        if args.dataset != 'imdb':
            save_dir += f"/{args.dataset}"
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
        do_train(args, model, train_dataloader, save_dir=save_dir)

        # Change eval dir
        args.model_dir = save_dir

    # Train model on the augmented training dataset
    if args.train_augmented:

        train_dataloader = create_augmented_dataloader(dataset)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        if args.model == 'gpt2':
            model.config.pad_token_id = model.config.eos_token_id
        model.to(device)
        save_dir = "./out_augmented"
        if args.model != 'bert':
            save_dir += f'/{args.model}'
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
        if args.dataset != 'imdb':
            save_dir += f"/{args.dataset}"
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
        do_train(args, model, train_dataloader, save_dir=save_dir)

        # Change eval dir
        args.model_dir = save_dir

    # Evaluate the trained model on the original test dataset
    if args.eval:
        out_file = os.path.basename(os.path.normpath(args.model_dir))
        out_file = out_file + "_original.txt"
        out_file = open(out_file, "w")

        score = do_eval(eval_dataloader, args.model_dir, out_file)
        print("Score: ", score)

        out_file.close()

    # Evaluate the trained model on the transformed test dataset
    if args.eval_transformed:
        out_file = os.path.basename(os.path.normpath(args.model_dir))
        out_file = out_file + "_transformed.txt"
        out_file = open(out_file, "w")

        eval_transformed_dataloader = create_transformed_dataloader(dataset, args.debug_transformation)
        score = do_eval(eval_transformed_dataloader, args.model_dir, out_file)
        print("Score: ", score)

        out_file.close()


