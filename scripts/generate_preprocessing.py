# Generates triples, other preprocessing data
#
# The outputs are put in sharedata/
#
# An argument must be passed, it is treated as a tag, so
#    'sharedata/butterfly-N%d-k%d-triples-%s' % (N, k, tag)

from math import log
from honeybadgermpc.mpc import generate_test_randoms, generate_test_triples
from apps.shuffle.butterfly_network import generate_random_shares
from time import time
import sys
from math import ceil

def go(tag):
    N, t, k = 16, 5, 4096
    NUM_SWITCHES = (k * int(log(k, 2)) ** 2) // 32
    print('start')
    generate_test_triples('sharedata/butterfly-N%d-k%d-triples-%s' % (N, k, tag), 2 * NUM_SWITCHES, N, t)
    print('triples')
    generate_random_shares('sharedata/butterfly-N%d-k%d-bits-%s' % (N, k, tag), NUM_SWITCHES, N, t)
    print('randoms')
    generate_test_randoms('sharedata/butterfly-N%d-k%d-inputs-%s' % (N, k, tag), k, N, t)
    print('test random')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python generate_preprocessing.py <tag>')
        print('used for xargs to generate preprocessing at once')
    tag = sys.argv[1]
    go(tag)
