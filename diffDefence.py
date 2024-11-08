import argparse
import json
import os
import time
import yaml

import torch
import pandas as pd

import reconstructphase as rp
from DDPM import DDPM
from attackpipeline import PGD_Attack, DF_Attack, JSMA_Attack, Zoo_Attack, \
    HSJ_Attack, CW_Attack, EN_Attack, BN_Attack, FGSM_Attack, PGD_Attack_CH
from autoencoder.model import TabularAutoEncoder
from classifiers.classifier import classifiers
from dataset import getData
from train import testadv

"""
    Main file
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_file = "all_results.yaml"

config = {
    "dataset": "UNSW",
    "classifier": "c_a3",  # name of the pretrained classifier
    "sub_classifier": None,  # clasifier used to create adv example for black box attack
    "typeA": None,  # type of the main classifier [ classifier_a or classifier_b]
    "typeB": None,  # type of the sub classifier [ classifier_a or classifier_b]
    "diffusion_name": "noordiffbal",  # Diffusion model name used to save the parameters
    "diffusion_step": 15,  # Diffusion step to generate the image
    "batch_size": 32,
    "test_size": 70000,  # Number of images to attack!
    "reconstruction_iter": 4,  # Iteration of the reconstruction phase
    "reconstruction_init": 5,  # Num. of restart of the reconstruction phase
    # Make sure that these are the same of the trained model!
    "beta_start": 0.00001,
    "beta_end": 0.02,
}

def main():

    parser = argparse.ArgumentParser(description="DiffDefence: main module!")

    parser.add_argument("--dataset", type=str, default="UNSW", help="dataset (UNSW)")
    parser.add_argument("--classifier", type=str, default="classifier_a", help="Name of the main classifier")
    parser.add_argument("--typeA", type=str, default="classifier_a", help="classifier_a or classifier_b")
    parser.add_argument("--sub_classifier", type=str, default=None, help="classifier_a or classifier_b")
    parser.add_argument("--typeB", type=str, default="classifier_b", help="classifier_a or classifier_b")
    parser.add_argument("--diffusion_name", type=str, default="noordiffbal", help="Diffusion model name")
    parser.add_argument("--diffusion_step", type=int, default=15, help="diffusion step to generate the image")
    parser.add_argument("--batch_size", type=int, default="32", help="Batch size")
    parser.add_argument(
        "--test_size", type=int, default=10000, help="number of image to attack, must be less than testset of dataset"
    )
    parser.add_argument("--reconstruction_iter", type=int, default=4, help="diffdefence reconstuction iteration")
    parser.add_argument("--reconstruction_init", type=int, default=5, help="diffdefence reconstuction initialization")
    parser.add_argument("--beta_start", type=float, default=0.0001, help="Beta start for diffusion model")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Beta end for diffusion model")

    args = parser.parse_args()

    # Load the configuration on the dict!
    config["dataset"] = args.dataset
    config["classifier"] = args.classifier
    config["sub_classifier"] = args.sub_classifier
    config["typeA"] = args.typeA
    config["typeB"] = args.typeB
    config["diffusion_name"] = args.diffusion_name
    config["diffusion_step"] = args.diffusion_step
    config["batch_size"] = args.batch_size
    config["test_size"] = args.test_size
    config["reconstruction_iter"] = args.reconstruction_iter
    config["reconstruction_init"] = args.reconstruction_init
    config["beta_start"] = args.beta_start
    config["beta_end"] = args.beta_end

    # Write configurations to YAML file
    with open(output_file, 'w') as yaml_file:
        yaml.dump(config, yaml_file)

    attacks = {
         #"PGD": PGD_Attack,
         "Deep Fool": DF_Attack,
         #"JSMA": JSMA_Attack,
         #"Zoo": Zoo_Attack,
         #"HSJ": HSJ_Attack,
         #"CW": CW_Attack
        # "EN": EN_Attack,
        #"BN": BN_Attack,
        #"FGSM": FGSM_Attack
        #"PGDCH": PGD_Attack_CH
    }

    for attackName in attacks:
        mainPipeline(config, attacks[attackName], attackName)


def getClassifier(config, submodel=False):
    """
    Function to get the pretrained classifier!
    """
    if not submodel:
        path = "./pretrained/" + config["dataset"] + "/" + config["classifier"] + ".pt"
        model = classifiers[config["typeA"]].to(device)

    if submodel:
        path = "./pretrained/" + config["dataset"] + "/" + config["sub_classifier"] + ".pt"
        model = classifiers[config["typeB"]].to(device)

    assert os.path.exists(path), f"Path: {path} do not exists"
    model.load_state_dict(torch.load(path))

    return model


def getGenerator(config):
    # Get Mask
    # mask = torch.load("data/mask.pt").bool().to(device)
    # active_dims = mask.sum().item()
    active_dims = 71
    # Get Autoencoder
    network = TabularAutoEncoder(active_dims, active_dims)
    network = network.to(device)

    # GET Diffusion
    model = DDPM(network, config["diffusion_step"], beta_start=config["beta_start"], beta_end=config["beta_end"])
    model.load_state_dict(torch.load(f"./pretrained/diffusion/{config['dataset']}/{config['diffusion_name']}.pt"))
    return model

def mainPipeline(config, attack, attackName):

    torch.manual_seed(0)

    # GET DIFFUSION MODEL
    diffusion = getGenerator(config)

    # GET CLASSIFIERS
    model = getClassifier(config)

    # For black box attack
    if config["sub_classifier"] is not None:
        smodel = getClassifier(config, submodel=True)

    # DATA TO ATTACK
    image_to_attack = getData(
        datasetname=config["dataset"], typedata="test", batch_size=config["batch_size"], test_size=config["test_size"]
    )

    # TRANSFORM IT IN ADVERSARIAL EXAMPLES - It returns adversarial examples and the original labels
    if config["sub_classifier"] is not None:
        # If Blackbox setting, use substitute model to create the adversarial examples
        adv_images, labels = attack(
            smodel, config["dataset"], config["classifier"], image_to_attack, config["batch_size"], config["typeA"]
        )
    else:
        adv_images, labels = attack(
            model, config["dataset"], config["classifier"], image_to_attack, config["batch_size"], config["typeB"]
        )

    # RECONSTRUCTION PHASE
    # mask = torch.load("data/mask.pt").bool().to(device)
    start_time = time.time()
    reconstructImages, _ = rp.reconstruction_pipeline(
        advdataset=adv_images,
        diffusionModel=diffusion,
        reciter=config["reconstruction_iter"],
        randiniti=config["reconstruction_init"],
        dim=71,
        # mask=mask,
    )
    finish_time = time.time() - start_time
    # Add masked data
    # fullReconstructions = torch.Tensor(size=[adv_images.shape[0], 204])
    # for i in range(adv_images.shape[0]):
    #     fullReconstructions[i, mask] = adv_images[i, mask]
    #     fullReconstructions[i, mask] = reconstructImages[i]

    print(f"---------------------{attackName}----------------------------------------")
    print(f"ACCURACY ON ADVERSARIAL IMAGES:{testadv(model, adv_images, labels, config['test_size']):.2f}%")
    print(f"ACCURACY ON RECONSTRUCTED IMAGES:{testadv(model, reconstructImages, labels, config['test_size']):.2f}%")
    print(f"Time to reconstruct one image {finish_time / config['test_size']:.4f}s")
    print("-------------------------------------------------------------------------")

    results = {
        "attack_name": attackName,
        "accuracy_adversarial": testadv(model, adv_images, labels, config['test_size']),
        "accuracy_reconstructed": testadv(model, reconstructImages, labels, config['test_size']),
        "time_per_image": finish_time / config['test_size']
    }

    # Write results to the output file in append mode
    with open(output_file, 'a') as outfile:
        outfile.write("------------------\n")
        yaml.dump(results, outfile)
        # Add a separator for each run
        outfile.write("------------------\n")

    print(f"Defence configuration results written to {output_file}")

    # Convert the tensor to a NumPy array
    reconstructImages_np = reconstructImages.cpu().numpy()  # Ensure it's on CPU

    # Create a DataFrame from the NumPy array
    df = pd.DataFrame(reconstructImages_np)

    # Save the DataFrame to a CSV file
    df.to_csv('reconstructed_images.csv', index=False)

# Usage in main():
if __name__ == "__main__":
    with open(output_file, 'w') as f:
        # Write YAML header
        f.write("---\n")

    main()
