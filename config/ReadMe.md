## Configuration Settings Explained

#### General

1) network_type : task that the network performs.  
   a) By classification network it is implied that image labels are known. So initial seeds can be directly selected.The crashes can also be directly detected for objective function.
   b) In Object detection network, we have many KPIs as measure of network's performance (exa. IoU, Precision, Recall etc.) The initial seeds are selected based on maximum IoU. The crash seeds are deteced whose mean IoU < 0.3.

2) num_sample : Number of images of a class to keep in initial test seeds
3) num_mutants : the number of mutants generated for each seed
4) max_iterations : maximum number of fuzz iterations
5) random : whether to adopt random testing strategy
6) seed_selection : strategy to select seeds from the seed queue, options - ['uniform', 'tensorfuzz', 'deeptest', 'prob']
7) fuzz_criteria : which coverage criteria to base fuzzing on, options -  ['nc', 'kmnc', 'nbc', 'snac', 'bknc', 'tknc', 'flann']
8) mutation_criteria : how to mutate images ['geometric']. For Object detection networks it can be from-['geometric', 'augmix']

9) classes : geometric transformation classes, numbers corresponding to indices of transformations list
10) transformation options in order (geometric): [translation, scale, shear, rotation, contrast, brightness, blur, pixel_change, image_noise]
classA : pixel value transformation, options - [7, 8]
classB : Affine transformation, options - [0, 1, 4, 5, 6]
NOTE: should not use rotation (2) and shear (3) unless network is trained for it

#### KIA/A2D2 Configuration:
1) data_path: Root directory where KIA/A2D2 datasets are stored locally
2) model_profile_path: Output location where model profile to be stored
3) splits_path: splits path for the train/val/test split declaration
4) mode: Dataset split to be used by fuzzer for fuzz testing 
5) init_seed_selection: strategy for selecting the initial seeds(currently: max_iou) (planned: precision, recall)
6) num_seed_eval: No. of seeds to be evaluated for identifying initial seeds (to be used only when evaluation of full seq. is not required)
7) num_seed_save: No. of seeds to be stored from the evaluated seeds based on their max_iou
8) num_seed_profile: No. of seeds from training set to be used to construct model profile