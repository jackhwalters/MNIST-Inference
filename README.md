Our customers have large amounts of images with digits that they want to classify using a ML model. The team has prepared the `setup-data.sh` and `train.py` scripts to come up with a classification model for digits. Have a look at that code and:
1. Train a model and save it to file;
2. Write a new Python script that loads the trained model and, given the `data_for_inference` folder, inspects all sub-folders and files and outputs the count of highest-likelihood predictions across all files - essentially how many occurrences of each digit are found in the target folder and subfolders.

Your inference script should scale to hundreds of thousands of files, and should be usable as part as a larger codebase. There is no need to worry about the results being "pretty", they only need to be human-readable, e.g.

```
$ python your_script.py --model mnist_cnn.pt --target data_for_inference
digit,count
0,29
4,1
```

You're free to change `setup-data.sh` and `train.py`, namely the data split and training setup, just document your reasoning. For convenience we include a `reset.sh` that you shouldn't need to change.

Sample usage of inference script:

```
$ python inference.py --model mnist_cnn.pt --target data_for_inference --batch-size 16 --image-types .png .jpg --output-file inference_results.csv

Inferencing batch 1 of 2
Inferencing batch 2 of 2
 digit  count
     0     11
     8      5
     6      4
     5      3
     2      3
     3      3
     4      1
```