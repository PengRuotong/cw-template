# Data Mining and Machine Learning Group Coursework
## Group Members

> [!IMPORTANT]
> Follow this format: `[Full Name] - @[GitHubUsername] - [HW Matriculation Number]`

1. `[Ruotong Peng] - @[PengRuotong] - [H00391709]` 
2. `[Hanjing Wang] - @[HanJingWang2024] - [H00391716]`
3. `[Jiawei Xu] - @[Healermm] - [H00391719]`
4. `[Yiyuan Lu] - @[Yiyuan Lu] - [H00391707]`
5. `[Lanting Huang] - @[Lanting Huang] - [H00391690]`

## Initial Project Proposal
### Source of Datasets
> [!IMPORTANT]
> Create a bullet list of the dataset(s) you used, their source with a link, and their licence. Also, include 2 specific examples from your dataset(s); present these nicely.

1. https://datasetsearch.research.google.com/search?src=3&query=music&docid=L2cvMTFsajY5dnRsMA%3D%3D
   https://creativecommons.org/licenses/by/4.0/
2. https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs
   https://opendatacommons.org/licenses/dbcl/1-0/
3. https://www.kaggle.com/code/andradaolteanu/work-w-audio-data-visualise-classify-recommend
   
### Milestones

> [!IMPORTANT]
> Create a bullet list of the key milestones for your project. Discuss these with your group. You may need to update these as the project progresses.

- **Dataset Selection and Presentation Preparation** (Thu 10/3/24 - Thu 10/10/24)  
  - Find datasets and organize information (Thu 10/3/24 - Mon 10/7/24)  
  - Create project plans, timelines, and PPT (Thu 10/3/24 - Tue 10/8/24)  
  - Organize information and prepare reports (Tue 10/8/24 - Thu 10/10/24)  

- **Data Analysis and Preprocessing** (Thu 10/10/24 - Tue 10/15/24)  
  - Perform data analysis and exploration (Thu 10/10/24 - Tue 10/15/24)  
  - Data preprocessing and cleaning (Thu 10/10/24 - Mon 10/14/24)  
  - Data feature analysis and visualization (Mon 10/14/24 - Tue 10/15/24)  

- **Clustering** (Wed 10/16/24 - Wed 10/23/24)  
  - Select appropriate clustering algorithms (Wed 10/16/24 - Tue 10/22/24)  
  - Evaluate algorithm performance and discuss (Wed 10/23/24)  

- **Baseline Training and Evaluation** (Thu 10/24/24 - Sat 11/2/24)  
  - Build prediction models (Thu 10/24/24 - Thu 10/31/24)  
  - Evaluate and compare models (Fri 11/1/24 - Sat 11/2/24)  

- **Neural Network Implementation** (Sun 11/3/24 - Mon 11/18/24)  
  - Implement neural network models (Sun 11/3/24 - Mon 11/11/24)  
  - Train and fine-tune neural networks (Tue 11/12/24 - Mon 11/18/24)  
  - Evaluate performance (Tue 11/19/24)  

- **Project Summary and Presentation** (Wed 11/20/24 - Tue 12/3/24)  
  - Write project summary and reports (Wed 11/20/24 - Mon 11/25/24)  
  - Prepare project display (Sat 11/23/24 - Tue 12/3/24)  


## Installing the project

> [!IMPORTANT]
> Provide instructions on how to install the project. This should include any dependencies that need to be installed.
Here’s a summary of the libraries used in the project:

1. **pandas** - Data manipulation and analysis.  
2. **matplotlib** - Data visualization.  
3. **seaborn** - Advanced data visualization.  
4. **numpy** - Numerical computations.  
5. **sklearn** - Machine learning models and data processing tools.  
6. **imbalanced-learn** - Data resampling methods (e.g., oversampling, undersampling).  
7. **nltk** - Natural language processing tools.  
8. **gensim** - Topic modeling and Word2Vec implementation.  
9. **torch (PyTorch)** - Deep learning framework.  
10. **wordcloud** - Visualization of text data as word clouds.  
```bash
```

## Data Preparation Pipeline

> [!IMPORTANT]
> Describe the data preparation pipeline. This should include how you will load the data, clean it, and preprocess it.

> [!TIP]
> Try to keep this as simple as possible, ideally with a single magic command that will run the entire pipeline consistently for everyone.


> [!WARNING]
> Do not blindly trust that the pipeline works. Verify that each invocation is identical by checking the output. For further checks, you can use methods such as calculating the MD5 checksum for files.

1. **Load Data**  
   Use `pandas` to read the raw dataset, retaining only the lyrics and song themes columns.

2. **Data Cleaning**  
   - Remove missing and duplicate values.  
   - Drop redundant and irrelevant features to focus on essential information.  

3. **Text Preprocessing**  
   - Use the `re` module to segment lyrics into sentences and remove punctuation.  
   - Remove stopwords using `nltk` to keep only meaningful words.  

4. **Data Balancing**  
   - Apply oversampling for minority classes by duplicating samples.  
   - Use undersampling for majority classes by removing excess samples.  

5. **Feature Extraction**  
   - Convert lyrics into 100-dimensional vectors using Word2Vec.  
   - Compute the mean word vector to represent each song as a numerical feature.  

6. **Data Visualization**  
   - Generate word clouds to display common words and their distribution across different song themes.  

7. **Clustering**  
   - Use the elbow method to determine the optimal number of clusters (k=5).  
   - Apply KMeans to cluster the lyrics vectors and generate cluster labels.  

8. **Save Data**  
   Save the processed data as a NumPy array for subsequent analysis and modeling.

## Coursework Requirements

> [!IMPORTANT]
> Include a short description (100 words max.) of each Coursework Requirement (R2-R5) and their location within your repository.

### R2. Data Analysis and Exploration
During the preprocessing phase, we conducted comprehensive processing of the dataset to ensure its suitability for machine learning model analysis. First, the downloaded dataset files were loaded into the Python environment, and data cleaning was performed, including checking for missing and duplicate values, removing redundant and irrelevant features, and retaining only the lyrics and song theme columns. Next, feature extraction, a critical step in text mining, was carried out. We simplified the text by applying sentence segmentation and stop-word removal to convert the lyrics into an analyzable format. Finally, random oversampling and undersampling methods were used to address data imbalance issues, ensuring that the model does not favor the majority class.
### R3. Clustering
This project utilizes natural language processing (NLP) techniques and the Word2Vec model to transform song lyrics into numerical features for machine learning. By segmenting lyrics and calculating the average word vector, each song is represented as a 100-dimensional vector. The vectors are converted into NumPy arrays for further analysis. KMeans clustering, a powerful unsupervised learning algorithm, is applied to group songs based on their lyrical features. Using the elbow method, the optimal number of clusters is determined to be 5, as indicated by the point where the within-cluster sum of squares (WCSS) begins to plateau.

### R4.	Baseline Training and Evaluation Experiments
The dataset was processed using multiple classification algorithms to ensure robust predictions. First, the Gaussian Naive Bayes algorithm was applied to split the data into training and testing sets, train the model, make predictions, and evaluate accuracy. A confusion matrix was used to assess classification performance. To address data bias, decision trees were utilized and subsequently improved into a random forest model. This approach constructed multiple decision trees, combining their predictions through voting or averaging to achieve the final output. Additionally, a k-Nearest Neighbour (k-NN) classifier was implemented, where the distance between a new data point and all training points was calculated. The k nearest neighbors were selected, and majority voting determined the category of the new data point.
### R5. Neural Networks
Convolutional Neural Networks (CNNs) build classifiers by leveraging one convolutional layer, one max pooling layer and 2 fully connected layers to extract spatial and hierarchical features from input data, which is word vectors here. 
The process involves convolving input data with learnable filters to create feature maps, followed by non-linear activation functions (e.g., ReLU) and pooling layers to reduce spatial dimensions while preserving essential features. Fully connected layers at the network’s end map these extracted features to specific class probabilities.

## Documentation

Weekly updates are kept in the `documentation/` directory.
