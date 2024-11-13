from src.markov import Markov
from time import time_ns


if __name__ == '__main__':
    # datasets: anekdoty(18+) hamlet ap_monkey
    m = Markov(file_path='datasets/ap_monkey.txt', log_file_path='logs/log_15_06_24.txt')
    # start = time_ns()
    chain = m.train(4)  # train ngrams
    # print((time_ns() - start) / 10 ** 9)
    inp = input("Type in some words (none to exit): ")
    while inp:
        m.generate_text(start_seq=inp, max_tokens=100, creativity=2)
        inp = input("Type in some words (none to exit): ")
