# Conversation Summarizer 

In this project I fine-tuned the pretrained Pegasus model from Hugging Face by training it on conversational data. I made this project mainly to consolidate what I had learnt about MLOps and DevOps.

# MLOps Workflow

- Data Ingestion and Basic Validation
- Data Transformation
- Model Training and Evaluation

The entire pipeline is automated using a Python script.

# Deployment on AWS 

- Docker Integration: Project uses Docker for containerization.
- Github Actions and CI/CD: Automates the process from pushing code to deployment on the cloud.
- AWS Elastic Container Registry: Store Docker container images.
- AWS EC2: Deployment onto an EC2 instance.
 
