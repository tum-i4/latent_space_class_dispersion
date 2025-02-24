# Creation of new mutation operators
New mutation operators can be put in any python file in the `mutation_ops` folder. The framework will search all the python files in this folder and will create the operators.

It looks for all classes that inherits from `TrainingMutation` or `ModelMutation` as well as all methods that do not start with an underscore '_', these are assumed to be training data mutation operators.

# Step 1: Dateset preprocessing
To handle datasets more efficiently, we only use them in preprocessed form everywhere. These are stored using pytorch as dictionary.

A stored data usually contains the following keys:
- img_xxx: pytorch float tensor of shape (N, C, H, W) containing the training/val/test/cc images
- lbl_xxx: pytorch uint8 tensor of shape (N,) containing ground-truth labels
- path_xxx: pytorch uint8 tensor of shape (N,) containing original ids
- num_classes: number of classes in the dataset

Please run "0_prepare_org_dataset.py", "1_prepare_fuzzing_dataset.py", "0_prepare_mix_dataset.py" to store original, corner case and combined data resepectively. 


# Step 2: Mutant generation
Now the mutants can be generated. For this you need a config file in which the model, data and the desired mutation operators are configured.

Simple example (configs/mnist/lenet_merged.ini):
```ini
[General]
model = models/mnist/lenet5/model.py
train = models/mnist/lenet5/train.py
eval = models/mnist/lenet5/sut.py
data = data/mnist/mnist_org_data_normalized.pth

[ChangeEpochs]
mutation = ChangeEpochs
epochs = 5

[RemoveSamples]
mutation = remove_samples
percentage = 0.5
```

This can then be passed to the script to generate the mutants:
```bash
python 4_mut_run.py mnist/lenet_merged.ini
```

This will create a new folder in `results/` based on the output name containing the raw_mutants.

# Step 3: Mutant training
These raw_mutants can then be trained using the `mut_train.py` script:
```bash
python 5_mut_train.py --num_trainings <n> results/<wanted_folder>
```

# Step 6: Calculate Accuracy and Centroids for LSCD calculations.
Use "6_calculate_accuracy.py" file to calculat accuracy using trained mutant models on datasets from "data" folder. This will create a folder `results/<wanted_folder>/evaln` that contains .parquet files for detailed analysis in further steps.

The same .parquet files are added with centroids values using "7_calculate_centroids.py" file. 


# Step 8: Final analysis
Now the generated test data can be finally evaluated to calculate LSCD and MS. Also, their correlation can be studing by running "8_lscd_ms_calculations.py".
