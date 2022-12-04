# Ad-hoc Federated Learning on Edge Devices

## Quick Start
To run experiments on decentralized vs centralized FL, run `decentralized_fl.py` with the command line arguments as specified in options.py
### Example Command
```
python decentralized_fl.py --num_share_devices 5 --comm_reliability 0.8 --model_share_strategy distance --run_name cifar_strat_dist_share_5 --seed 1
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
└── downloader.py                # Small script to speed up downloading trial data as csv's from Tensorboard
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