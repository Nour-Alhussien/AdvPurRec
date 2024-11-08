# AdvPurRec: Strengthening Network Intrusion Detection with Diffusion Model Reconstruction Against Adversarial Attacks
 [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2309.03702)


Abstract: *The ongoing race between attackers and defenders in cybersecurity hinges on who makes the first move. Defenders have a significant advantage if they can anticipate and counteract attacks preemptively. However, if attackers are aware of the defenders' strategies, meaningful defense becomes exceedingly challenging. In this paper, we propose a reactive defense technique that adapts to the presence of attacks and mitigates their effects. We introduce an adversarial purification defense technique that leverages the capabilities of a diffusion denoising probabilistic model to eliminate adversarial noise. Through training the purifier and classifier independently on clean examples, our defense remains robust against unseen attacks, making it an agnostic defense method. We rigorously evaluate our defense technique using network intrusion datasets, demonstrating its superiority over other state-of-the-art defense techniques in terms of both effectiveness and efficiency. Our results demonstrate the potential of this method to significantly enhance the resilience of network intrusion detection systems against adversarial threats. To ensure the reproducibility of our results, we have made our implementation publicly available.

## Requirements

Before running the code, install conda, and requirements and activate the environment env

```
conda create --name env --file requirements.txt
conda activate env
```

# Test Instructions

Make sure to run the setup file to create all the necessary folders!

```
$python setup.py 
```

# Create the dataset

First, you must create the dataset. Add all the CSV files to the data folder then run the following command.

```bash
$python create_dataset_iot.py
```

## Train the classifiers

First of all, you must train the classifiers to test AdvPurRec. The training can be logged with wandb, set --wandb='run' if you want to log.

```
$python train.py --dataset='UNSW' --classifier='classifier_a' --model_title='c_a'
```


To train adversarial trained classifier, set --adv_train.

```
$python train.py --dataset='UNSW' --adv_train=True --classifier='classifier_a' --model_title='c_a_adv'
```

Check the train.py file to set other configuration parameters.

## Train the Diffusion Model

Afterwards, train the Diffusion model. Check the DDPM.py file to set other configuration parameters.

```
$python DDPM.py --dataset='UNSW' --model_title='diff2'
```


## Test DiffDefence

Finally, test DiffDefence!

```
$python diffDefence.py --dataset='UNSW' --classifier='c_a' --typeA='classifier_a' --diffusion_name='diff' --test_size=1000
```


- If you want to test adversarial trained model against adversarial attacks

```
$python diffDefence.py --dataset='MNIST' --classifier='c_a_adv' --typeA='classifier_a' --diffusion_name='diff' --test_size=1000
```


## Citation

If you use this codebase, or otherwise found our work valuable, please cite:

```


```
