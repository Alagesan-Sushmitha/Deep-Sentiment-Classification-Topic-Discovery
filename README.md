# Deep-Sentiment-Classification and Topic-Discovery with NLP using LSTM Recurrent Network approach

# Question
How can health services be improved using sentiment classifications based on the remarks made by Covid 19 patients between January 2020 and March 2020 on a social media platform like Reddit from across the world?

# Importance

* Having a clear understanding of the comments on a social media platform about COVID-19 can be vital in providing valuable insights into the experiences and opinions of people regarding the pandemic and their access to healthcare services. 
* This information can help the health services department to understand and address the challenges faced by people such as the availability of medical supplies, the quality of healthcare services, and the accessibility of healthcare facilities.
* By analyzing the sentiments expressed by people on a social media platform like Reddit, healthcare providers can identify areas that need improvement and take proactive measures to address them. 
* This can ultimately lead to the expedited and improved delivery of health services, which is particularly important during a global health crisis like the Covid 19 pandemic.

# Stakeholders
#### Healthcare providers: 
The primary stakeholder of this question that we are addressing here is the healthcare providers. They typically gain insights into the challenges faced by the people during the pandemic in a way that can help them improve their services and respond to the needs of their patients. 
#### Patients and Families: 
Patients and families are the second important stakeholder as they are the ones who are directly impacted by the quality of health services. By analyzing user comments on social media platforms, healthcare providers can gain important information about a patient’s opinion and seek ways to improve their care. 
#### Public health agencies:
Public health agencies are the ones responsible for developing policies and programs that adhere to public health issues. By analyzing user comments on social media platforms, they can gain insights into the challenges faced by people during the pandemic. 
#### Researchers and academics:
Researchers and academics make use of sentiment analysis to conduct research and develop new insights into the experiences of people during the pandemic. They can use this to publish research papers and contribute to the academic literature on the topic.

# Importance of NLP
Here are a bunch of reasons why NLP is the best to respond to the question above:
*	NLP can generally handle large amounts of unstructured data, such as user comments on social media platforms, and analyze them in a relatively short time.
*	NLP can effectively perform sentiment analysis to identify and categorize opinions and emotions expressed in user comments related to COVID 19
* NLP can process text in multiple languages making it possible to analyze user comments related to COVID-19 from different parts of the world. This can further provide insights into how the pandemic is affecting people in different regions and help in developing appropriate responses. 
* NLP can be easily integrated with other technologies such as machine learning and data visualization to provide a comprehensive understanding of user comments related to COVID-19.

# Assessing the process and outcome of the project
The process and outcome of the LSTM model for COVID-19 comments can be assessed using various evaluation metrics. Firstly, the model's performance on the training and validation sets can be monitored during the training process by analyzing the loss and accuracy metrics. Secondly, the model's performance can be evaluated on a separate test dataset that was not used during training. This can be done by calculating the model's accuracy, and loss percentage. Additionally, we can perform cross-validation to assess the model's performance more accurately. Finally, we can compare the results of our model with other state-of-the-art models and techniques used for sentiment analysis on similar datasets to evaluate its effectiveness.

# Data Acquisition
Data acquisition is the process of collecting and gathering raw data from various sources and transforming it into a format that can be used for analysis or other applications.
### Identify the data source
 The methodology for collecting data sources for the deliverable will involve several steps to ensure the data is appropriate. First, we have already identified sources of COVID-19 patient comments, posted on Reddit from Kaggle. There was a total of 5,63,079 comments extracted, assessed, and used for this experiment.
The dataset was collected between January 20, 2020, and March 19, 2020.
We have then evaluated the data to ensure they contain relevant comments from COVID-19 patients that are suitable for sentiment analysis and topic modeling. Some of the NLP techniques that have been incorporated are Topic Modelling – LDA & Gibbs Sampling, Sentiment Analysis / determining the sentiment polarity, Machine learning algorithms for sentiment analysis, Long-Short Term Model (LSTM).

# Data Cleaning
Steps taken to determine whether the data should be considered sensitive.
* Sensitive data is any information that, if revealed, could harm, or embarrass people, as it relates to COVID-19 patient comments. Examples of sensitive information include personally identifiable information (PII), names of persons, financial information about the cost of medical care in a certain hospital, and any other information that can be used to identify a person.
*	Evaluating the data's possible hazards, such as the danger of injury or embarrassment to certain people if the data were revealed. 
*	Data protection procedures are implemented to safeguard the data based on the risk assessment. For example, the data may be anonymized, access restrictions may be put in place, or sensitive data fields may be encrypted.
* Legal or ethical requirements linked with the collection and management of sensitive data are necessary.

  ![image](https://github.com/Alagesan-Sushmitha/Deep-Sentiment-Classification-Topic-Discovery/assets/137837229/1bccc6d3-e125-48fd-9705-312257bd7fcf)
  
  ![image](https://github.com/Alagesan-Sushmitha/Deep-Sentiment-Classification-Topic-Discovery/assets/137837229/7fce5ed5-0545-4f92-9bda-ead627b57fc9)

#### Data Processing

![image](https://github.com/Alagesan-Sushmitha/Deep-Sentiment-Classification-Topic-Discovery/assets/137837229/244cb1fc-5e56-4a4e-b55c-33ce80670ea3)

![image](https://github.com/Alagesan-Sushmitha/Deep-Sentiment-Classification-Topic-Discovery/assets/137837229/c44e435b-ec60-428e-be29-bf2952c77663)

![image](https://github.com/Alagesan-Sushmitha/Deep-Sentiment-Classification-Topic-Discovery/assets/137837229/f72223b0-c558-4e65-9319-c5314446f688)

![image](https://github.com/Alagesan-Sushmitha/Deep-Sentiment-Classification-Topic-Discovery/assets/137837229/6a102f1e-5d3b-4c16-86dd-fb72a4ed7f4f)

# Data Modelling
### Model Identification
The model we have chosen is the Deep learning Neural network model – LSTM (Long-Short Term Model).

### Why this model is optimal?
For natural language processing (NLP) applications like sentiment classification and topic discovery, the LSTM (Long Short-Term Memory) neural network model is a viable option since it is particularly well-suited for evaluating sequential input, such as text.

### Strengths of this model
#### Flexibility in handling different kinds of input data:
With several types of input features, like word embeddings or character-level representations, LSTMs can handle input sequences of varying lengths.
#### Improved performance:
On a variety of natural languages processing tasks, such as sentiment classification, language modeling, and machine translation, it has been proven that LSTMs can perform at the cutting edge of technology.
#### Ability to handle sequential data: 
Since LSTMs are made to capture long-term dependencies in sequential data, they are excellent for applications involving natural language processing, including speech recognition, sentiment analysis, and language translation.
#### Memory Retention: 
The issue of vanishing gradients, which can be a serious problem with conventional recurrent neural networks, is prevented by the capacity of LSTMs to selectively store and forget information from prior time steps.
#### Robustness to noise & stop words: 
For natural language processing tasks where there may be a lot of noisy or irrelevant data, LSTMs are resilient to input noise, which is significant.

![image](https://github.com/Alagesan-Sushmitha/Deep-Sentiment-Classification-Topic-Discovery/assets/137837229/99242b1d-8ed3-4d03-b98e-6f6378d706a5)

![image](https://github.com/Alagesan-Sushmitha/Deep-Sentiment-Classification-Topic-Discovery/assets/137837229/34fb5c4c-5390-40ad-9b95-c10ae6def656)

![image](https://github.com/Alagesan-Sushmitha/Deep-Sentiment-Classification-Topic-Discovery/assets/137837229/f4f673f5-aa42-4197-918e-1bee90b9a9b9)

![image](https://github.com/Alagesan-Sushmitha/Deep-Sentiment-Classification-Topic-Discovery/assets/137837229/27f18f0a-065d-4d9b-9c87-35d915d877c5)

![image](https://github.com/Alagesan-Sushmitha/Deep-Sentiment-Classification-Topic-Discovery/assets/137837229/47c7e01f-e7c0-4fa2-b923-5f6a41f7d5b9)

![image](https://github.com/Alagesan-Sushmitha/Deep-Sentiment-Classification-Topic-Discovery/assets/137837229/14dd624a-5bde-45c4-a3b6-a7b4dfafb2c3)

# Data Analysis 





