"""
    Assuming all folds have been trained to the same number of epochs, this code goes through the train, validation
    and test set and records the accuracy and balanced accuracy across all folds and then calculates the mean and
    standard error over these folds. The output will be put in the "Metrics" directory.

    ├── Project
    │   ├── Graphs
    │   │   ├── SN0
    │   │   │   ├── NodeFiles
    │   │   │   ├── EdgeFiles
    │   │   ├── SN1
    │   │   │   ├── NodeFiles
    │   │   │   ├── EdgeFiles
    │   ├── LogFiles
    │   ├── Models
    │   ├── Metrics  <---- output will be in here

"""


