# 🤖 Hate Speech Classification Using AWS SageMaker Pipeline and HuggingFace Transformers

**Description**  

Hate speech is a very common thing in this day and age. As we know, the growth of social media makes it possible for the people to spread their opinions and thoughts to be heard by the world. This can be a good thing or a bad thing depending on the context. Then, if hate speech is spreading with the growth of technology, why don’t we counter them also with technology. 

**Objective** 

In this project, we aim to deploy a high-performance Hate Speech Classifier that can be trained again in the future with the new version of dataset.   

**Conclusion**  

We used AWS SageMaker to develop the model and also deploy them. The deployment makes it possible for the model to do prediction using API. We used the Pipeline feature from AWS SageMaker so we can deploy a new version of the model in the future automatically if we had a new dataset or to make variations of the model using different parameters. A fine-tuned BERT Transformers is chosen as the classifier because it is a high-performance sequence classifier. 

**Data Source** 

Thomas Davidson, Dana Warmsley, Michael Macy, and Ingmar Weber. 2017. "Automated Hate Speech Detection and the Problem of Offensive Language." Proceedings of the 11th International Conference on Web and Social Media (ICWSM). Original paper link :
https://aaai.org/ocs/index.php/ICWSM/ICWSM17/paper/view/1566
