import sys
import time
from collections import OrderedDict
from operator import itemgetter


def main():
    # Check for correct arguments length
    if len(sys.argv) != 3:
        print("Expected n_gram_analyzer.py <n-gram size> <file-name>")
        sys.exit()
    start_time = time.time()
    NGRAM_SIZE = int(sys.argv[1])
    FILE = sys.argv[2]

    n_gram_list = get_gram_frequencies(NGRAM_SIZE, FILE)
    end_time = time.time()
    print(n_gram_list)
    print("Total time:", end_time-start_time)


def get_gram_frequencies(n_gram_size, file, allowed_bytes=None):
    dict = {}
    with open(file, "rb") as analyzeFile:
        while True:
            bytes = analyzeFile.read(n_gram_size)
            if len(bytes) < n_gram_size:
                break
            if not allowed_bytes or (allowed_bytes and bytes in allowed_bytes):
                add_to_dict(dict, bytes)
            analyzeFile.seek(n_gram_size-1, 1)
    # return sorted(dict.items(),key=itemgetter(1), reverse=True)
    return dict


def add_to_dict(dict, key):
    if key in dict:
        dict[key] += 1
    else:
        dict[key] = 1


if __name__ == '__main__':
    main()
