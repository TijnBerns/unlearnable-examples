# unlearnable-examples

This repository contains the the code created for the bachelor thesis Exploring Unlearnable Examples.
The thesis explores the resistance of error-minimizing noise against adversarial training and data augmentation.

### How to use to program

Train in a natural manner on a clean CIFAR-10 dataset:  
```/path_to_python/python3 main.py --todo train_nat --dataset cifar10```

Generate sample-wise noise for CIFAR-10:  
```/path_to_python/python3 main.py --todo train_sample_wise --dataset cifar10 --delta_path path_to_noise_file --save sample_wise```

Train in a natural manner on a perturbed CIFAR-10 dataset:  
```/path_to_python/python3 main.py --todo train_nat --dataset poison_cifar10 --delta_path  path_to_noise_file --noise name_of_noise_file```

For more help on how to use the program type:  
```/path_to_python/python3 main.py --help```
