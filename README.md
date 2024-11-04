# USmorph
## 1. Introduction
`USmorph` is a machine learning framework that combines unsupervised and supervised algorithms to classify galaxy morphologies. Unlike traditional machine learning algorithms, `USmorph` does not rely on pre-labeled data and addresses the sensitivity of traditional algorithms to image orientation through an adaptive polar coordinate transformation (APCT) method. If you have used this algorithm, please cite [Song et al. 2024](https://doi.org/10.3847/1538-4365/ad434f). For detailed information about the `USmorph` algorithm, you can refer to [Zhou et al. 2022](https://doi.org/10.3847/1538-3881/ac4245), [Fang et al. 2023](https://doi.org/10.3847/1538-3881/aca1a6), [Dai et al. 2023](https://doi.org/10.3847/1538-4365/ace69e), and [Song et al. 2024](https://doi.org/10.3847/1538-4365/ad434f)

## 2. Installation instruction
### 2.1 Clone this repo
You can clone this repository to your local machine using the following command:
```git clone https://github.com/jsong-astro/USmorph.git ```
Alternatively, you can also download the relevant `.zip` file from the [GitHub](https://github.com/jsong-astro/USmorph) website.

### 2.2 Set up your environment
We strongly recommend that you should create a new virtual environment for `USmorph` to minimize conflicts with dependencies for other packages. Additionally, since our program can be mainly divided into unsupervised and supervised parts, we recommend creating separate virtual environments for each of these parts.

**UML Method**
``` 
conda create -n uml_env python=3.7
conda activate uml_env
pip install -r uml_requirement.txt
```
**SML Method**
```
conda create -n sml_env python=3.7
conda activate sml_env
pip install -r sml_requirement.txt
```
## 3. Usage
### 3.1 Image Preparation
- Enter the `/USmorph/UML/1_cae_noise` folder and place your files into the `raw_fits` folder. Then run `python readimg_fits.py`to convert the images to PNG format, which will be stored in the `fit_img` folder.
- Configure `config.py` to set the model parameters, including image dimensions and convolution kernel size.
- Run `python MAIN.py --train True` to train the model for auto encoding.
- Run `python MAIN.py --test True` to obtain the denoised images in the `cae_img/` directory.
- Run `python polar.py` to convert the images in `cae_img/` to polar coordinate images, and output them to the `polar_img/` directory.

### 3.2 UML Method
#### 3.2.1 ENCODE
- Enter the `/USmorph/UML/2_cae_encode` folder and configure the `config.py` file to set the model parameters, including image dimensions and convolution kernel size.
- Run `python make_pkl.py` to convert the target images into pickle format .txt files.
- Run `python MAIN.py --train True` to train the model for auto encoding
- Run `python MAIN.py --encode True` to obtain the `encode_result` file, which contains the encoded data.
#### 3.2.2 Cluster
- Enter `3_cae_class` folder and copy the `encode_result` generated in your previous step to this folder. 
- configure the `config.py` file to set how many groups you want to cluster, and choose the cluster algorithm you want to use
- run `python main.py` to get the cluster result for each cluster algorithm in `./result/` folder
- run `python get_final_result.py` to get the final cluster result in `./result_final` folder
- visual inspection the final cluster result and classify the into groups you need, in this step, you should generate `result.txt` to point out the categories that each group belongs to.
- run `python make_dataset.py` to generate the classification result to `./dataset` package, this can be used in the following SML method.

### 3.3 SML Method
If you selected more than one clustering algorithm in the previous steps, some samples may not be classified due to inconsistent voting. In this case, you will need to use the UML method to classify this batch of samples. If you only selected one clustering algorithm, you can skip this step.
Here are the detailed steps:
1.  **enter `./USmorph/SML/` folder,  configure `conf/global_settings.py` to set the input data dimensions, the interval of epochs for storing weights, the interval steps for testing (default is 100, meaning accuracy is tested every 100 steps), the dataset address, and the test set address.**
2. **To train the model, execute the command:**
```
	python train.py -net model_name -gpu -b 32 
	-net model_name: the model you used, e.g., googlenet
	-gpu           : omit this if the device has no GPU; defaults to CPU
	-b 32          : the batch size, which can be adjusted based on server performance; reducing it will 
					 decrease memory usage, and the value should be a power of 2
	
	# For example, you would input:
	python train.py -net googlenet -gpu -b 32
```
3. **To test the model, execute the command:**
```
	python test.py -net model_name -weight path_to_checkpoint_file -b 1000
	-------------------------------------------------------------------
	-net model_name: the model you used, e.g., googlenet
	-weight path_to_checkpoint_file: the path you save your check point file
	-b 32: the batch size
	
	# For example, you would input:
	python test.py -net googlenet -weight checkpoint/my_checkpoint_file -b 1000
```
4. **After training is complete, use the model's stored weights to classify unlabeled images by executing the command:**
```
	python getlabel.py -net <model_name> -weight path_to_checkpoint_file -b 1000
```
