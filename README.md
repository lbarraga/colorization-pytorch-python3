# Interactive Deep Colorization in PyTorch

> note: This fork fixes some issues the original repository had with Python 3.
> 
> All credits go to [Richard Zhang](https://richzhang.github.io/)\*, [Jun-Yan Zhu](http://people.csail.mit.edu/junyanz/)\*, 
  [Phillip Isola](http://people.eecs.berkeley.edu/~isola/), [Xinyang Geng](http://young-geng.xyz/), 
  Angela S. Lin, Tianhe Yu, and [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/).  

[Project Page](https://richzhang.github.io/ideepcolor/) |  [Paper](https://arxiv.org/abs/1705.02999) | [Video](https://youtu.be/eL5ilZgM89Q) | [Talk](https://www.youtube.com/watch?v=rp5LUSbdsys) | [UI code](https://github.com/junyanz/interactive-deep-colorization/)

## Fast start

1. clone this repo and create a virtual environment:
  ```bash
  git clone https://github.com/lbarraga/colorization-pytorch-python3.git
  cd colorization-pytorch-python3
  python3 -m venv venv
  source venv/bin/activate
  ```

2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

3. Download the [ILSVRC 2012 dataset](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php)(training images (728MB) and validation images (6.3GB)). extract the training images and put them in a folder <folder>/train. Extract the validation images and put them in a folder <folder>/val.
4. Prepare the dataset:
  ```bash
  python make_ilsvrc_dataset.py --in_path <folder>
  ```

5. Download a pretrained model:
  ```bash
  bash pretrained_models/download_siggraph_model.sh
  ```

6. Test the model:
```bash
python test.py --name siggraph_caffemodel --how_many 10 --mask_cent 0
```
    
7. A results folder will be created with the results of the test. You can visualize the results by opening the index.html file in the results folder.

## Testing the model

You can test the model on the validation data by running the following command:
```bash
python test_sweep.py --name siggraph_caffemodel --mask_cent 0
```

in the code of test_sweep.py you can set the `how_many` flag, that has been hardcoded to something different.

in `./checkpoints/siggraph_caffemodel` you will find a csv with the mean and std for each number of points, 
along with the plot of the test.

To visually compare the results/performance of the different models, run:
```bash
python plot.py
```
