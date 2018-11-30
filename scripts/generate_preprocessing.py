from math import log
from honeybadgermpc.mpc import generate_test_randoms, generate_test_triples
from apps.shuffle.butterfly_network import generate_random_shares
from time import time

N, t, k = 50, 16, 32
NUM_SWITCHES = k * int(log(k, 2)) ** 2
print('start')
generate_test_triples('sharedata/butterfly-N%d-k%d-triples' % (N, k), 2 * NUM_SWITCHES, N, t)
print('triples')
generate_random_shares('sharedata/butterfly-N%d-k%d-bits' % (N, k), NUM_SWITCHES, N, t)
print('randoms')
generate_test_randoms('sharedata/butterfly-N%d-k%d-inputs' % (N, k), k, N, t)
print('test random')
