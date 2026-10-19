# Notes

## Method
* What method of formulating AI projects should be used?

## Use Case Selection
* What problem do I want to solve for my users? What are my goals and measures of success?
* What baseline solution are there to compare against?
* Is ML the right solution? How well does it need to perform?
* What is the definition of good?
* Has the problem been solved before?

## Learning Paradigm and Model Selection
* What learning paradigm is required? Unsupervised, supervised, or reinforcement learning? Semi-supervised or weakly supervised?
* What error metric will be used? What is your target value? How will you manage the accuracy coverage tradeoff?
* What is the model's degree of outlier sensitivity?
* What model do I need? fixed size vectors: MLPs, data with topological structure: CNNs, time series: start with LSTM, GRU, or transformer

## Data Availability and Quality
* How much training compute is required?
* Do you have enough data? Do you know where it comes from?
* Are the labels accurate?
* How much data do I need? What is the cost of data collection?

## Computing Requirements
* Where will the model physically train and infer?
* How much inference compute is required? What are the inference performance specifications? Latency, queries per second, RMA, interpretability, retraining time?
* How much training compute is required? What is the training schedule and budget?

## Budget and Schedule
* Training and inference costs?
* Development timeline and rhythm?
* Team required
* Budget

