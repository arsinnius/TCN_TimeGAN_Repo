# **TCN_TimeGAN_Repo**
Adapted from the excellent paper by Jinsung Yoon, Daniel Jarrett, and Mihaela van der Schaar:  
[Time-series Generative Adversarial Networks](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks),  
Neural Information Processing Systems (NeurIPS), 2019.

- Last updated Date: April 24th 2020
- [Original code](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/alg/timegan/) author: Jinsung Yoon (jsyoon0823@gmail.com)
<br><br>
# **Description**
The model uses Tensorflow 2. In the original timegan paper, the autoencoder was built using an RNN. This project replaces the RNN with the TCN-AE autoencoder described by Thill et al. (2021) in "Temporal convolutional autoencoder for unsupervised anomaly detection in time series."

The modified model was tested on two different time series - financial and climate. Using the default settings, TCN-AE outperformed the RNN on the financial time series. However it didn't perform very well on the climate data. The belief is that the performance on both series could be vastly improved if the parameters of the TCN-AE were individually optimized for each time series type.
<br><br>
# **Getting Started**
The code is executed in Jupyter notebooks. It was written and tested using Google Colab. At the top of each notebook is four alternating cells – markdown, code, markdown, code. The first markdown cell is “Colab Code” and the following code cell mounts the google colab drive. The second markdown is “Start Folder” and the following code cell changes  the directory to the repo directory. The cells can easily be relabeled and rewritten to accommodate other platforms.
<br><br>
# **Installing**
Under the markdown header cell “Installs” there are installs applicable to the colab environment. These can be modified to accommodate other environments or removed completely if not required.
<br><br>
# **Repo Structure**

Three folders are added to the standard .git folder and github files – `experiment_08`, `experiment_09` and `src`.

![Repo_dir.png](https://drive.google.com/uc?id=1JYYiXTRsJJxVsP_dJ5Jr9GSZnbxBDDP-)
<figcaption></figcaption>

The `src` directory is shown below.

![src_dir.png](https://drive.google.com/uc?id=1KjAUkORKK8kpgZpsMYaJHqac-Vjei-xq)
<figcaption></figcaption>

The `src` folder contains the files necessary to run the experiments. 
1. `params_08` – parameter settings for `Financial_TS.ipynb` (experiment 08).
2. `params_09` – parameter settings for `Climate_TS.ipynb` (experiment 09).
3. `tcn_ae.py` – code for the TCN autoencode.
4. `train_fns.py` – training functions for the component models of TCN_TimeGAN.

`experiment_08` contains the synthetic financial time series experiment and `experiment_09`, the climate time series experiment. The `experiment_##` folders are organized similarly. For example, the following shows `experiment_08`.

![experiment_08_dir.png](https://drive.google.com/uc?id=1s999CvdjmERAsAlDn6AJ3r4KorJ9XpUc)
<figcaption></figcaption>

Notice that `experiment_08` contains three Jupyter notebooks:

1.   `Financial_TS.ipynb` – Creates the synthesizer that will be used to
generate the synthetic time series.
2.   `Eval_08.ipynb` – runs tests to evaluate the quality of the synthetic time series produced
3. `New_Synthetic_Generation_08.ipynb` – Uses the files created by `Financial_TS.ipynb` to generate more synthetic time series
    

In addition, it contains the following folders:
1. `autoencoder` – contains the autoencoder model saved in Tensorflow 2 format.
2. `data` – contains raw time series data or may be empty if the data is read in directly, for example by `pandas_data_reader`.
3. `eval` – contains the evaluation results produced by `eval_08.ipynb`.
4. `events` – contains the events recorded by the logger `tf.summary.create_file_writer`.
5. `samples` – contains synthetic data samples created by `Financial_TS.ipynb`
6. `supervisor` – contains the supervisor model saved in Tensorflow 2 format.    
7. `autoencoder` – contains the autoencoder model saved in Tensorflow 2 format.

The src folder contains the files necessary to run the experiments. The directory is shown below.

![src_dir.png](https://drive.google.com/uc?id=1KjAUkORKK8kpgZpsMYaJHqac-Vjei-xq)
<figcaption></figcaption>


1. `params_08` – parameter settings for `Financial_TS.ipynb`.
2. `params_09` – parameter settings for `Climate_TS.ipynb`.
3. `tcn_ae.py` – code for the TCN autoencode.
4. `train_fns.py` – training functions for the component models of TCN_TimeGAN.
<br><br>
# **Executing the Code**

The code is executed in Jupyter notebooks. The notebooks are structured to be tutorial and are meant to be used in order. After loading and initializing the notebook , the user can either run all cells and go to the end to see the results or step through cell-by-cell to examine model structures, check on tensor shapes, watch values change, etc. experiment_08 is illustrative.
 
a. `Financial_TS.ipynb` 

**Outline**
1. Title
2. Colab Code
3. Installs
4. Imports & Settings
5. Experiment Path
6. Prepare Data
7. TimeGAN Components
8. TimeGAN Training
    * Phase 1: Autoencoder Training
    * Phase 2: Supervised Training
    * Joint Training
9. Generate Synthetic Data

At the end random samples of real and synthetic time series are printed on the same graph so the user can see if the synthetic data “looks right.” The samples are also saved in the `samples` folder.
<br><br>
b. `eval_08.ipynb`

**Outline**
1. Title
2. Colab Code
3. Sample to be Evaluated
4. Imports
5. Parameters
6. Experiment Path
7. Load Real and Synthetic Data
8. Prepare Sample
9. Visualization in 2D: A Qualitative Assessment of Diversity
10. Time Series Classification: A Quantitative Assessment of Fidelity
11. Train on Synthetic, Test on Real: Assessing Usefulness

After generating synthetic samples, the user may evaluate the quality of the synthetic data. There are three kinds of evaluations done:
1. **Visualization in 2D** – the notebook plots the data using two different dimensionality reduction algorithms – principal components analysis (PCA) and t-distributed Stochastic Neighbor Embedding (t-SNE). PCA is inherently linear. t_SNE is a nonlinear dimensionality reduction technique. That means the algorithm allows the separation of data that cannot be separated by a straight line.
2. **Time Series Classification** – the notebook creates a simple classifier, calculates the metrics accuracy and AUC (area under the curve) and compares the results for the real and synthetic time series. The synthetic data should fool the classifier 50% of the time. That is, the classifier should be no better than random guessing.
3. **Train on Synthetic, Test on Real (TSTR)** - TSTR vs. TRTR (Train on Real, Test on Real). The notebook creates a regression model that tries to predict the last time step for the six stock prices. It plots the Mean Absolute Error (MAE) using TSTR and TRTR and compares the results. The synthetic data is useful if the performance using TSTR is equal to or better than using TRTR. Results are saved in the `evals` folder.

**NOTE!! These calculations are for illustration only!!!** The notebook calculates tests using a single sample. A rigorous evaluation would involve performing the tests on many (thousands of) combined samples. The notebook can be readily modified to accomplish this.
<br><br>
c. `New_Synthetic_Generation_08.ipynb`

**Outline**
1. Title
2. Colab Code
3. Installs
4. Samples to be Generated
5. Imports
6. Parameters
7. Get Data
8. Get Synthesizer
9. Prepare Data
10. Plot Sample Series

This notebook can be used to generate new synthetic data using the models and files created by  `Financial_TS.ipynb`. Following the markdown cell “Installs”, is the markdown “Samples to be Generated” followed by a code cell containing two variables which the user must assign values – `n_init` (starting sample number) and `n_plots` (number of samples). Suppose the user assigns the values 11 and 5 respectively. Then the notebook will generate 5 samples (`sample_11` to `sample_15`) and place them in the samples folder. To avoid overwriting samples currently in that folder, the use must make sure that the numbers don’t overlap. For example, one could use 1-100 for one set, 101-200 for another and so on. 

Clearly, the notebook in its present form is for illustrative purposes. In reality, after perhaps many rounds of generating, evaluating  and tweaking the code, the user may wish to create thousands (millions) of synthetic time series to be used for training an AI/ML model. This notebook would have to be modified. Luckily, the Jupyter notebook format is flexible. For example, rather than storing samples in the samples folder, they could be written to an `h5` file and every *n*th sample printed to indicate progress.
<br><br>
# **Starting a New Experiment**

Suppose a user wants to create a new experiment – `experiment_20`. Creating `experiment_20` would begin by creating a `params_20.py` file in the `src` folder. The parameter file is structured as follows:

1. imports
2. experiment number
3. Data Parameters
4. Training Parameters
5. Network Parameters
6. Generic Loss Functions
7. Optimizers
8. Pooling Parameters

The Data Parameters section from `params_08.py` and `params_09.py` are shown below.

![params_08_data](https://drive.google.com/uc?id=1CyzxZhdbMcAgFW1SXZoPAKnZEPD1tfMF)


![params_09_data](https://drive.google.com/uc?id=1ZmOkmUv2KTQcH7hB3mjk-WzUrYLc1rav)

The difference in the data sections reflects the difference in the data. The stock data was daily data read in from Yahoo using `pandas_data_reader`. The climate data was collected from the [Historical Palmer Drought Indices](https://www.ncei.noaa.gov/access/monitoring/historical-palmers/) and stored in a `.csv` file. 

The next step would be to modify `Financial_TS.ipynb` (say) to accommodate the new data. The Experiment Path section of the notebook will use the parameter `experiment` to automatically create an `experiment_20` folder in the repo and populate it with the required sub-folders.

Altering the training or tweaking the networks can be done by changing the appropriate parameters. If new parameters are required to, for example, tune the TCN, they can be easily added to `params_20.py`.
<br><br>
# **Issues**

1. The TCN was tested using default settings. It outperformed on the financial data but didn’t perform well on the climate data. Can the parameters of the TCN be optimized separately for the two types of time series?
2. More generally, can a user-guide be written to explain how to optimize TCN parameters based on time series characteristics for *any* time series?
<br><br>
# **Author(s)**
Norbert Pierre – arsinnius@yahoo.com
<br><br>
# **License**
This project is licensed under the MIT License - see [MIT License](https://opensource.org/licenses/MIT) for details 
<br><br>
# **Acknowledgements**

1. [Machine Learning for Algorithmic Trading](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715/ref=sr_1_1?crid=2EOY5VIDNELLR&keywords=Jansen%2C+%22Machine+Learning%22&qid=1668352381&s=books&sprefix=jansen%2C+machine+learning+%2Cstripbooks%2C139&sr=1-1), 2nd edition by Stefan Jansen

2. Thill, Konen, Wang and Back (2021), [Temporal convolutional autoencoder for unsupervised anomaly detection in time series](https://www.sciencedirect.com/science/article/abs/pii/S1568494621006724)

3. Yoon, Jarrett and van der Schaar (2019), [Time Series Generative Adversarial Networks](https://proceedings.neurips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf)
