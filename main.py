from src.markov import Markov
from datetime import date


def logname(date: date):
    date_str = '_'.join(map(str, [date.day, date.month, date.year]))
    return 'log_' + date_str + '.txt'


if __name__ == '__main__':
    # datasets: anekdoty(18+) hamlet ap_monkey
    
    m = Markov(file_path='datasets/ap_monkey.txt', 
               log_file_path='logs/' + logname(date.today()),
               creativity=2)

    chain = m.train(4)  # train ngrams

    inp = input("Type in some words (none to exit): ")
    while inp:
        m.generate_text(start_seq=inp, max_tokens=100)
        inp = input("Type in some words (none to exit): ")

