from collections import defaultdict
import string
import random
import time

class Markov():
    def __init__(self, file_path, log_file_path=None, creativity=0, dynamic_order=1):
        self.file_path = file_path

        self.creativity = creativity
        ''' 0 - picking the first picked most popular token; 
            1 - picking any of the most popular tokens; 
            2 - picking tokens randomly with weights;
            3 - picking tokens randomly with uniform distribution.
        '''

        self.dynamic_order = dynamic_order
        ''' when matching ngrams with defined order can't be found we try finding shorter ngrams '''

        self.trained: bool = False
        ''' whether the model is trained '''

        # Logging: 
        self.logging: bool = True if log_file_path else False
        if self.logging:
            self.log_filepath = log_file_path
            try:
                with open(log_file_path, 'a') as logf:
                    pass
            except FileNotFoundError as err:
                print(f"File {log_file_path} not found. Logs won't be saved. {err}")
                self.logging = False
            except OSError:
                print(f"OS error occured trying to open {log_file_path}. Logs won't be saved.")
                self.logging = False
            except Exception:
                print(f"Unexpected error trying to open {log_file_path}. Logs won't be saved. ")
                self.logging = False
            else:
                self.log_linen = 1  # number of log lines
                self.add_log(f"Object {self} initialized.", separate=True)
        
        self.text = self.remove_punctuations(self.get_text())
    
    def add_log(self, text=None, separate=False):
        if not self.logging:
            return -1
        with open(self.log_filepath, 'a') as logf:
            if separate:
                logf.write("\n-------------------------------------\n\n")
            
            if text:
                logf.write(f"{self.log_linen}. {time.asctime(time.localtime())[4:]} :: {text}\n")
                self.log_linen += 1
        return 0

    def get_text(self):
        '''
        This function will read the input file and return the text associated to the file line by line in a list
        '''
        text = []
        with open(self.file_path) as f:
            for line in f:
                text.append(line)
        return ' '.join(text)
    
    @staticmethod
    def remove_punctuations(text):
        '''
        Given a string of text this function will return the same input text without any punctuations
        '''
        return text.translate(str.maketrans('','', string.punctuation))
    
    @staticmethod
    def towords(text: str, lower: bool=True):
        text = text.lower() if lower else text
        return text.split()

    @staticmethod
    def tokens_to_indices(tokens: list | str, trans_table: dict[str: int]) -> list | str:
        try:
            if isinstance(tokens, str):  # if single string is passed
                result = trans_table[tokens]
            else:
                result = list(map(lambda token: trans_table[token], tokens))
        except KeyError as err:
            print(f"Error: \tThe token (word) '{err.args[0]}' is not present in the training set.\n"
                  "\tTry using different words or reducing order of Markov chain (or changing training text)")
            return None
        return result

    def train(self, n: int=1, logging: bool=True):
        '''
        This function will take a block of text and map each sequence of <order> words as a key with
        value being the next ngram. All words (tokens) are presented as indices to save memory

        args:
            n (int) : Length of ngrams (order)
            stdlog (bool) : whether the info messages are needed (like Successfully trained)
        '''
        print("Training started. The bigger dataset is the more time it will take (~1 sec/10 MB)")
        self.order = n  # saving the order

        # Tokenizing: split the input text into tokens
        # this time the tokens are individual words separated by spaces (or \t, \n, \f)
        tokens = self.towords(self.text)

        if logging:
            self.add_log(f"Training started with order={n}, file '{self.file_path}'")
        
        self.token_to_ind = {token: i for i, token in enumerate(set(tokens))}  # collecting all the unique tokens
        self.ind_to_token = {i: token for token, i in self.token_to_ind.items()}

        self.ngrams_dict = defaultdict(lambda: [[], 0])
        
        ''' example: {(12, 13, 2): [[2, 1, 687, 0, 99], 9],
                      , ...}
            where 12, 13, 2, 1, 687, 0, 99 - indices indicating tokens in self.ind_to_token array;
                  the (2, 1, 687, 0, 99) array contains all of the possible next tokens;
                  9 - amount the ngram (12, 13, 2) appeared in the training data (self.text)
        '''
        ind_tokens = tuple(self.tokens_to_indices(tokens, self.token_to_ind))
        
        for i in range(len(tokens) - n):
            gram = ind_tokens[i:i + n]
            next_token = ind_tokens[i+n]
            if gram in self.ngrams_dict:
                if next_token in self.ngrams_dict[gram][0]:
                    next_gram = ind_tokens[i + 1:i + n + 1]
                    self.ngrams_dict[next_gram][1] += 1 # incrementing next_gram's counter
                else:
                    self.ngrams_dict[gram][0].append(next_token)
            else:  # basically only works for the first gram
                self.ngrams_dict[gram] = [[next_token], 1]

        if logging:
            self.add_log('Training ended.')
        self.trained = True

        return self.ngrams_dict
    
    def closest_match_ngram(self, gram: tuple[int], logging=True) -> tuple[tuple, list[list, int]]:
        ''' Function for picking ngram matching the given one '''
        if len(gram) >= self.order:
            ngram = gram[-self.order:]
            if ngram in self.ngrams_dict:
                result = ngram, self.ngrams_dict[ngram]
            else:
                result = None
        elif len(gram) < self.order:
            matching_ngrams: list[tuple, int] = []  # ngram and their popularity
            for key in self.ngrams_dict:
                if key[-len(gram):] == gram:
                    matching_ngrams.append(key, self.ngrams_dict[key][1])
            picked_ngram = self.creative_pick(matching_ngrams)
            result = picked_ngram, self.ngrams_dict[picked_ngram] if picked_ngram else None
        if logging:
            self.add_log(f"Function <closest_match_ngram> ended. Picked ngram: {result[0] if result else None}")
        return result

    def creative_pick(self, ngrams_n_pops: list[tuple, int], logging=True) -> list[int] | None:
        ''' Function for picking ngram based on set creativity setting '''
        if logging:
            self.add_log(f"Function <creative_pick> started. ngrams_n_pops[:5]: {ngrams_n_pops[:5]}")
        if not ngrams_n_pops:
            if logging:
                self.add_log(f"Function <creative_pick> ended. ngrams_n_pops list was empty")
            return None
        
        match self.creativity:
            case 0:  # 0 - picking the first picked most popular token 
                ngram = max(ngrams_n_pops, key=lambda ngram_n_pop: ngram_n_pop[1])[0]
            case 1:  # 1 - picking any of the most popular tokens
                # get popularity of the most popular token:
                greatest_pop = ngrams_n_pops[0][1]
                most_pops = list(filter(lambda gram_n_poplr: gram_n_poplr[1] == greatest_pop, ngrams_n_pops))
                ngram = random.choice(most_pops)[0]
            case 2:  # 2 - picking tokens randomly with probabilty weights
                counts = map(lambda ngram_n_count: ngram_n_count[1], ngrams_n_pops)
                ngram = random.choices(ngrams_n_pops, weights=counts)[0][0]
            case 3:  # 3 - picking tokens randomly with uniform distribution
                ngram = random.choice(ngrams_n_pops)[0]
            case _:
                raise ValueError(f"Unknown creativity value: {self.creativity}")
            
        if logging:
            self.add_log(f"Function <creative_pick> ended. Picked ngram: {ngram}({' '.join(map(lambda i: self.ind_to_token[i], ngram))})")
        return ngram

    def crpick_nexttoken(self, cur_ngram: list, nexttokens: list, logging=True) -> tuple[int]:
        ''' Function for picking next token based on set creativity setting.
            Returns ngram with new token '''
        if logging:
            self.add_log(f"Function <crpick_nexttoken> started. Current ngram: {cur_ngram}, Next tokens: {nexttokens}")
        if not nexttokens:
            if logging:
                self.add_log(f"Function <crpick_nexttoken> ended. nexttokens list was empty")
            return None
        
        ngrams_n_pops: list[tuple[tuple, int]] = []  # list of new ngrams and their popularities

        for index_token in nexttokens:
            gram = tuple(list(cur_ngram) + [index_token])
            ngram_data = self.closest_match_ngram(gram)  # closest ngram with required order
            if ngram_data:
                popularity = ngram_data[1][1]
                ngrams_n_pops.append((ngram_data[0], popularity))
        picked_ngram = self.creative_pick(ngrams_n_pops)
        
        if logging:
            nextt = picked_ngram[-1] if picked_ngram else None
            self.add_log(f"Function <crpick_nexttoken> ended. Next token: {nextt}({self.ind_to_token[nextt] if nextt else None})")
        return picked_ngram

    def predict_next_ngram(self, ind_tokens: list[int], logging=True):
        if logging:
            self.add_log(f"Function <predict_next_ngram> started. ind_tokens={ind_tokens}.")

        ind_tokens = tuple(ind_tokens)  # to be able to compare with dict keys which are tuples

        # forming list of ngrams that match input tokens
        possible_ngrams = []
        last_n_tokens = len(ind_tokens)
        for gram, properties in self.ngrams_dict.items():
            if ind_tokens == gram[-last_n_tokens:]:
                possible_ngrams.append((gram, properties[1]))  # ((11, 1151), 12)
        
        if not possible_ngrams and self.dynamic_order:
            # trying to find matching ngram with less last tokens
            last_n_tokens -= 1
            while not possible_ngrams and last_n_tokens >= 1:
                for gram, properties in self.ngrams_dict.items():
                    if ind_tokens[-last_n_tokens:] == gram[-last_n_tokens:]:
                        possible_ngrams.append((gram, properties[1]))
                last_n_tokens -= 1

        if not possible_ngrams:  # if still couldn't find
            if logging:
                self.add_log(f"Function <predict_next_ngram> ended. No possible predictions found.")
            return None
        
        # pick ngram based on creativity setting to predict next token:
        ngram = self.creative_pick(possible_ngrams)
        next_tokens = self.ngrams_dict[ngram][0]
        # find best new ngram:
        best_ngram = self.crpick_nexttoken(ngram, next_tokens)
        
        if logging:
            self.add_log(f"Function <predict_next_ngram> ended. Next ngram: {best_ngram}")
        return best_ngram
    
    def generate_text(self, start_seq, max_tokens=10, creativity=0, dynamic_order=1, logging=True):
        '''
        Given the staring text and the number of tokens, this function will allow you to predict the
        next up to <max_tokens> tokens in the sequence
        
        args:
            start_seq (String) : The text to start prediction from
            max_tokens (int) : maximum amount of tokens to generate
        
        example:
            markov_model.generate_text(start_seq='I do not know what to write here', max_tokens=5)
        '''
        if logging:
            self.add_log(f"Function <generate_text> started. startseq='{start_seq}', max_tokens={max_tokens}, "
                         f"creativity={creativity}, dynamic_order={creativity}")
        
        if not self.trained:
            if logging:
                self.add_log(f"Function <generate_text> ended early. The model is not trained")
            print("The model is not trained. Use train() method first. Aborting.")
            return None

        self.creativity = creativity
        self.dynamic_order = dynamic_order

        # Tokenizing:
        tokens: list[str] = Markov.towords(Markov.remove_punctuations(start_seq))
        ind_tokens = self.tokens_to_indices(tokens, self.token_to_ind)  # indexes of tokens (ids)
        if ind_tokens is None:
            if logging:
                self.add_log(f"Function <generate_text> ended early. Training data is incomplete for current prompt")
            return None
        
        gentokens_list = [] # list of predicted tokens
        print(f"\nGenerated text: {start_seq} ", end='')

        if len(ind_tokens) > self.order:
            ind_tokens = ind_tokens[-self.order:]  # using last <self.order> tokens
        
        while len(gentokens_list) < max_tokens:
            next_ngram = self.predict_next_ngram(ind_tokens)
            if not next_ngram:
                break
            else:
                ind_tokens = next_ngram
                next_token = self.ind_to_token[next_ngram[-1]]
                gentokens_list.append(next_token)
                print(f"{next_token} ", end="")

        if not gentokens_list:
            print("! No text could be generated: possibly training data is lacking (try using dynamic order if you haven't already) !")
        if logging:
            self.add_log(f"Function <generate_text> ended. {len(gentokens_list)} tokens generated",
                         separate=True)
        
        print()
        return gentokens_list
