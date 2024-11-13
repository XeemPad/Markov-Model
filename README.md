# Markov Chain NLP model
Practical part of the Student Scientific Research on Markov Chains in
Natural Language Proceccsing

BMSTU, Robotics and automatization Department, 2024

## Program settings
To configure the Markov model, in main.py you can change such variables as:

- file_path - path to dataset (text file)

- log_file_path (Optional) - desired path to logs

- creativity - determines whether the model should pick the most probable
words (statistically defined by dataset and ngrams_len) or be more loose
(for specific values go to src/markov.py -> Markov.__init__.py)

- ngrams_len - how much previous tokens (words) should the model take into account
when generating the next word

- max_tokens - how much tokens (words) should be generated based on input