# How to use this pipeline

This repo gives you an opportunity to train your own CycleGAN just with 2 strings in terminal!

## Datasets

At first you should download a dataset for your model.

###### Warning!
 
At this moment just one dataset is supported by my script (summer2winter_yosemite).

To download it, just run following command in your terminal:

    ./scripts/get_dataset.sh
 
## Train

After downloading the dataset you should just run a single script with python:

    python train_models.py \
        -ds1 ds/summer2winter_yosemite/trainA \ 
        -ds2 ds/summer2winter_yosemite/trainB
    
And after some time you will get the result!

To customize training, you can change some variables for models with keywords in this command:

`-sh`: can be `True` or `False`. Sets whether the pairs in datasets are shuffled or not (Positively affected on my tests)

`-gopt`: can be `Adam` or `AdamW`. Sets the optimizer for generators

`-dopt`: can be `Adam` or `AdamW`. Sets the optimizer for discriminators

`-gsc`: can be `default` or `step10warmup`. Sets the scheduler for generators. 

Default one works like it described in original paper on CycleGAN, but with variable step after which learning rate 
will be decreased.

`-dsc`: can be `default` or `step10warmup`. Sets the scheduler for discriminators. 

Default one works like it described in original paper on CycleGan, but with variable step after which learning rate 
will be decreased.

`-stp`: sets epoch for default scheduler, when the learning rate will start decreasing

`-glr`: sets learning rate of generators

`-dlr`: sets learning rate of discriminators

`-ep`: sets amount of epochs that 

`-th`: sets loss threshold for updating discriminators (recommended: 2)

`-is`: sets image size that is used to train models