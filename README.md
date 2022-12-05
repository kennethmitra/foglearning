# Ad-hoc Federated Learning on Edge Devices

## Quick Start
### Install dependencies
Install dependencies with `pip install -r requirements.txt` in order to run our code.

### Running Experiments
To run experiments on decentralized vs centralized FL, run `decentralized_fl.py` with the command line arguments as specified in options.py
#### Example Command
```
python decentralized_fl.py --num_share_devices 5 --comm_reliability 0.8 --model_share_strategy distance --run_name cifar_strat_dist_share_5 --seed 1
```

## Usage Help
```
usage: decentralized_fl.py [-h] [--run_name RUN_NAME] [--dataset DATASET]
                           [--model MODEL] [--input_channels INPUT_CHANNELS]
                           [--output_channels OUTPUT_CHANNELS]
                           [--shuffle_dataset SHUFFLE_DATASET] [--iid IID]
                           [--batch_size BATCH_SIZE]
                           [--num_local_update NUM_LOCAL_UPDATE]
                           [--num_share_rounds NUM_SHARE_ROUNDS]
                           [--num_share_devices NUM_SHARE_DEVICES]
                           [--fed_avg FED_AVG]
                           [--num_total_rounds NUM_TOTAL_ROUNDS] [--lr LR]
                           [--lr_decay LR_DECAY]
                           [--lr_decay_epoch LR_DECAY_EPOCH]
                           [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY]
                           [--verbose VERBOSE] [--num_devices NUM_DEVICES]
                           [--comm_reliability COMM_RELIABILITY]
                           [--model_share_strategy {random,distance,ring}]
                           [--seed SEED] [--use_gpu USE_GPU]
                           [--global_model GLOBAL_MODEL]
                           [--local_model LOCAL_MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --run_name RUN_NAME   string for the name of the run
  --dataset DATASET     name of the dataset: mnist, cifar10
  --model MODEL         name of model. mnist: logistic, lenet; cifar10:
                        cnn_tutorial, cnn_complex
  --input_channels INPUT_CHANNELS
                        input channels. mnist:1, cifar10 :3
  --output_channels OUTPUT_CHANNELS
                        output channels
  --shuffle_dataset SHUFFLE_DATASET
                        shuffle the order of the dataset for each client
  --iid IID             shuffle the order of the dataset for each client
  --batch_size BATCH_SIZE
                        batch size when trained on client
  --num_local_update NUM_LOCAL_UPDATE
                        number of local gradient update steps (tau_1)
  --num_share_rounds NUM_SHARE_ROUNDS
                        number of weight sharing steps (tau_2)
  --num_share_devices NUM_SHARE_DEVICES
                        Number of devices to share with at each share round
  --fed_avg FED_AVG     Perform federate averaging. If true, no decentralized
                        training will be don
  --num_total_rounds NUM_TOTAL_ROUNDS
                        Number of total (train + share) rounds to perform
  --lr LR               learning rate of the SGD when trained on client
  --lr_decay LR_DECAY   lr decay rate
  --lr_decay_epoch LR_DECAY_EPOCH
                        lr decay epoch
  --momentum MOMENTUM   SGD momentum
  --weight_decay WEIGHT_DECAY
                        The weight decay rate
  --verbose VERBOSE     verbose for print progress bar
  --num_devices NUM_DEVICES
                        number of all available devices
  --comm_reliability COMM_RELIABILITY
                        Fraction of time communication works
  --model_share_strategy {random,distance,ring}
                        random or distance or ring
  --seed SEED           random seed (defaul: 1)
  --use_gpu USE_GPU     Use gpu or not
  --global_model GLOBAL_MODEL
  --local_model LOCAL_MODEL
```

## File Structure

### Main Directory Code
```
.
├── decentralized_fl.py          # Main script used to run experiments on decentralized federated learning (Logs to runs/ with tensorboard)
├── device.py                    # Contains device class with code for sending / receiving weight updates
├── models/                      # Directory containing NN models used in project
├── options.py                   # Defines command line argument structure and default values
├── auto_run.py                  # Script that automatically runs trials continuously and saves progress in case of interruption
├── graphing.py                  # Code used to generate figures from the results directory
├── average.py                   # Contains function to average sets of model weights together
├── requirements.txt             # List of dependencies and versions required for running this code
├── downloader.py                # Small script to speed up downloading trial data as csv's from Tensorboard
├── M1/                          # Directory containing code forked from https://github.com/LuminLiu/HierFL used to generate M1 results (Hierarchical FL vs Centralized FL)
└── .gitignore                   # List of files to ignore when commiting to GitHub
```

### Results Folder
```
results/
├───cifar_comm_reliability_exp   # Data on experiments with communication reliability as the variable
├───cifar_noniid_exp             # Data on experiments using nonIID datasets among devices
├───cifar_num_share_exp          # Data on experiments using cifar10 with num_share_devices as the variable
├───cifar_strat_compare          # Data on experiments testing random vs distance based communication strategy
├───figures                      # Directory containing generated graphs and charts
└───mnist_num_share_exp          # Data on experiments using mnist with num_share_devices as the variable
```