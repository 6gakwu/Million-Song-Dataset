# README
Purpose
This README provides the required instructions for running the implementation, the necessary libraries, and the structure of the submitted code folder.

### How to Run the Code
Step 1: Ensure the dataset files are in the data/ directory

data/

    kaggle_songs.txt
    kaggle_users.txt
    kaggle_visible_evaluation_triplets.txt

Step 2: Install required Python libraries

The implementation uses Pythjon 3.10+ (tested on Python 3.13).

Install dependencies:

> pip install numpy pandas matplotlib

Step 3: Run the main program

From the project root directory:

> python main.py

### Optional Configuration
All hyperparameters are defined inside main.py. You may modify them directly. Examples:

> K = 50        #Top-K recommendation

> small_eval_users = eval_users[:500]       #evaluation subset size

> max_iters = 500       # SA iteration count

> T_start = 1.0         #SA initial temp

### Required Libraries and Versions
numpy -> Version 1.26+

pandas -> Version 2.0+

matplotlib -> Version 3.7+

### File Structure

DS8001_project_code_Ogakwu_Philemon -> /data/graphs/src main.py readme.md

/data -> kaggle_songs.txt kaggle_users.txt kaggle_visible_evaluation_triples.txt

/graphs -> hitrate.png map@K.png precision@K.png recall@K.png

/src -> content_based.py data_loader.py evaluation.py item_cf.py popularity.py simulated_annealing.py splitter.py user_cf.py
