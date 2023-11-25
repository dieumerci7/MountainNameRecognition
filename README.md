Named Entity Recognition for Mountain Names
This project focuses on training a Named Entity Recognition (NER) model to identify mountain names within text. The following components are included:

1. Dataset Creation
The dataset used for this project is derived from the DFKI-SLT/few-nerd dataset. Relevant rows were extracted, and the data was relabeled. Dataset statistics and preprocessing details can be found in the dataset.ipynb Jupyter notebook.

The balanced dataset is located in the balanced_datasets directory, divided into training, validation, and test sets.

2. Model Training
For NER, the dslim/bert-large-NER model and tokenizer were used. Training can be initiated using the following command:

bash
Copy code
python train.py --output_dir ./fine_tune_bert_output --learning_rate 2e-5 --num_train_epochs 0.01
3. Model Inference
Inference on new text samples can be performed using the following command:

bash
Copy code
python inference.py --text "I visited the Classification Mountains when I was a child."
The results and demo examples can be found in the demo.ipynb Jupyter notebook.

4. Model Weights
The trained model weights can be accessed via the following Hugging Face Model Hub link: dieumerci/mountain-recognition-ner

Project Structure
balanced_datasets: Contains the balanced training, validation, and test datasets.
fine_tune_bert_output: Output directory for the trained model.
dataset.ipynb: Jupyter notebook with dataset creation and statistics.
train.py: Python script for model training.
inference.py: Python script for model inference.
demo.ipynb: Jupyter notebook with demo examples.
Feel free to explore the notebooks and scripts for more details on the dataset, training, and inference processes. If you encounter any issues or have questions, please reach out!