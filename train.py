import torch
from torch import optim
import wandb
from classifiers.classifier import *
from tqdm import tqdm
from dataset import *
from torch.utils.data import DataLoader, TensorDataset

import argparse

# for adv training
from attackpipeline import *

# Hyperparameters for the training pipeline
hyperparameters = {
    "LR": 1e-2,
    "EPOCHS": 1,
    "BATCH_SIZE": 128,
    "MOMENTUM": 0.9,
    "CLASSES": 2,
    "DATASET": "UNSW",
    "CLASSIFIER": "classifier_b",
    "MODEL_TITLE": "name",
    "LOG": "disabled",
}

# check for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    print("Entered main function")

    parser = argparse.ArgumentParser(description="DiffDefence: Train classifier module")

    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--dataset", type=str, default="UNSW", help="dataset (UNSW)")
    parser.add_argument("--epochs", type=int, default="1", help="Training epochs")
    parser.add_argument("--batch_size", type=int, default="128", help="Batch size")
    parser.add_argument("--momentum", type=float, default="0.9", help="momentum")
    parser.add_argument("--classes", type=int, default="2", help="dataset class")
    parser.add_argument("--model_title", type=str, default="modelname", help="Model name")
    parser.add_argument("--log", type=str, default="disabled", help="Wandb logging")
    parser.add_argument("--classifier", type=str, default="classifier_a", help="classifier_a or classifier_b")
    parser.add_argument("--adv_train", type=bool, default=False, help="Adversarial training")

    args = parser.parse_args()

    hyperparameters["LR"] = args.lr
    hyperparameters["DATASET"] = args.dataset
    hyperparameters["EPOCHS"] = args.epochs
    hyperparameters["BATCH_SIZE"] = args.batch_size
    hyperparameters["MOMENTUM"] = args.momentum
    hyperparameters["CLASSES"] = args.classes
    hyperparameters["MODEL_TITLE"] = args.model_title
    hyperparameters["LOG"] = args.log
    hyperparameters["CLASSIFIER"] = args.classifier

    print(f"Hyperparameters: {hyperparameters}")

    print("Loading data...")
    trainloader, testloader = getData(
        datasetname=hyperparameters["DATASET"], typedata="both", batch_size=hyperparameters["BATCH_SIZE"]
    )
    print("Data loaded successfully")

    model_pipeline(
        classifierName=hyperparameters["CLASSIFIER"],
        datasetname=hyperparameters["DATASET"],
        trainloader=trainloader,
        testloader=testloader,
        adv_train=args.adv_train,
    )


def model_pipeline(classifierName, datasetname, trainloader, testloader, adv_train):
    print("Entered model_pipeline")

    with wandb.init(project="classifier-diffusion-defense", config=hyperparameters, mode=hyperparameters["LOG"]):
        # access all HPs through wandb.config
        config = wandb.config

        # make the model, data and optimization problem
        model, criterion, optimizer = create(config, classifierName)
        print("Model, criterion, and optimizer created")

        # train the model
        if adv_train is True:
            print(f"Adversarial training on {config['CLASSIFIER']}")
            adversarial_train(model, trainloader, criterion, optimizer, config, classifierName, datasetname, testloader)
        else:
            print(f"Training {config['CLASSIFIER']}")
            train(model, trainloader, criterion, optimizer, config, classifierName, datasetname, testloader)

        # test the model
        print(f"Accuracy test: {test(model, testloader)}%")

    return model


def create(config, classifierName):
    print("Entered create function")

    # Create a model
    model = classifiers[classifierName].to(device)

    # Create the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LR, momentum=config.MOMENTUM)

    print("Model, criterion, and optimizer initialized")

    return model, criterion, optimizer


def adversarial_train(model, trainloader, criterion, optimizer, config, classifierName, datasetname, testloader):
    print("Entered adversarial_train function")
    r"""
    Method that implement adversarial training
    """
    if wandb.run is not None:
        wandb.watch(model, criterion, log="all", log_freq=10)

    example_ct, batch_ct = 0, 0

    for epoch in range(config.EPOCHS):  # loop over the dataset multiple times
        pbar = tqdm(trainloader, leave=False)
        for _, (images, labels) in enumerate(pbar):
            print(f"Epoch {epoch}, Batch {_}")

            # create adversarial samples
            dl_to_adv = DataLoader(TensorDataset(images, labels), batch_size=images.shape[0])
            input, l = FGSM_Attack_CH(
                submodel=model, datasetname=datasetname, classifiername=classifierName, testset=dl_to_adv, batchSize=32
            )
            input, images, l, labels = input.to(device), images.to(device), l.to(device), labels.to(device)

            images = torch.cat((images, input), 0)
            labels = torch.cat((labels, l), 0)

            loss = train_batch(images, labels, model, optimizer, criterion)

            example_ct += len(images)
            batch_ct += 1

            pbar.set_postfix(MSE=loss.item())

        torch.save(model.state_dict(), f"./pretrained/{config['DATASET']}/{config['MODEL_TITLE']}.pt")
        train_log(loss, example_ct, epoch)
    return model


def train(model, trainloader, criterion, optimizer, config, classifierName, datasetname, testloader):
    print("Entered train function")

    # telling wand to watch
    if wandb.run is not None:
        wandb.watch(model, criterion, log="all", log_freq=10)

    example_ct, batch_ct = 0, 0

    for epoch in range(config.EPOCHS):  # loop over the dataset multiple times
        pbar = tqdm(trainloader, leave=False)
        for _, (images, labels) in enumerate(pbar):
            print(f"Epoch {epoch}, Batch {_}")

            loss = train_batch(images, labels, model, optimizer, criterion)

            example_ct += len(images)
            batch_ct += 1

            pbar.set_postfix(MSE=loss.item())

        torch.save(model.state_dict(), f"./pretrained/{config['DATASET']}/{config['MODEL_TITLE']}.pt")

        train_log(loss, example_ct, epoch)
    return model


def train_batch(images, labels, model, optimizer, criterion):
    print("Entered train_batch function")

    # insert data into cuda if available
    images, labels = images.to(device), labels.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # backward pass
    loss.backward()

    # step with optimizer
    optimizer.step()

    return loss


def train_log(loss, example_ct, epoch):
    print(f"Logging: Epoch {epoch}, Loss {loss}")

    if wandb.run is not None:
        wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)


def test(model, test_loader):
    print("Entered test function")
    model.eval()

    with torch.no_grad():
        correct, total = 0, 0
        for batch, labels in test_loader:
            batch, labels = batch.to(device), labels.to(device)
            outputs = model(batch)
            output_class = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (output_class == labels).sum().item()

    accuracy = (correct / total) * 100
    print(f"Test accuracy: {accuracy}%")
    return accuracy


def testadv(model_, images_, labels_, n_):
    print("Entered testadv function")

    model_.eval()

    with torch.no_grad():
        correct, total = 0, n_
        for i in range(n_):
            images, labels = images_[i].to(device), labels_[i].to(device)
            outputs = model_(images[None, :])

            predicated = torch.argmax(outputs, dim=1)

            correct += (predicated == labels).sum().item()

        if wandb.run is not None:
            wandb.log({"test_accuracy": correct / total})

        accuracy = 100 * correct / total
        print(f"Adversarial test accuracy: {accuracy}%")
        return accuracy


if __name__ == "__main__":
    print("Starting main")
    main()
