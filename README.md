# Named Entity Recognition for Mountain Names
This project focuses on training a Named Entity Recognition (NER) model to identify mountain names within text. The following components are included:

# 1. Dataset Creation
The dataset used for this project is derived from the [DFKI-SLT/few-nerd](https://huggingface.co/datasets/DFKI-SLT/few-nerd) dataset. Relevant rows were extracted, and the data was relabeled. Dataset statistics and preprocessing details can be found in the `dataset.ipynb` Jupyter notebook.

The balanced dataset is located in the `balanced_datasets` directory, divided into training, validation, and test sets.

# 2. Model Training
For NER, the [dslim/bert-large-NER](https://huggingface.co/dslim/bert-large-NER) model and tokenizer were used. Training can be initiated using the following command:

`python train.py --output_dir [output_dir] --learning_rate [learning_rate] --num_train_epochs [num_train_epochs]`

(The default arguments are *2e-5* for `learning_rate` and *5* for `num_train_epochs`).

# 3. Model Inference
Inference on new text samples can be performed using the following command:

`python inference.py --text [text]`

The results and demo examples can be found in the `demo.ipynb` Jupyter notebook.

# 4. Model Weights
The trained model weights can be accessed via the following Hugging Face Model Hub link: [dieumerci/mountain-recognition-ner](https://huggingface.co/dieumerci/mountain-recognition-ner/commit/7f81a566f20e6c58c77c6ed0e459ae6c0a94356f)

# Project Structure
- `balanced_datasets`: Contains the balanced training, validation, and test datasets.
- `dataset.ipynb`: Jupyter notebook with dataset creation and statistics.
- `train.py`: Python script for model training.
- `inference.py`: Python script for model inference.
- `demo.ipynb`: Jupyter notebook with demo examples.
- `requirements.txt`: the requirements, call using
  `pip install -r requirements.txt`

Feel free to explore the notebooks and scripts for more details on the dataset, training, and inference processes. If you encounter any issues or have questions, please reach out!
