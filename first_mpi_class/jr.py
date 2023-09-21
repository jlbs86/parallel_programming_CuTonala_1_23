from mpi4py import MPI
import numpy as np

if (__name__ == '__main__'):
  comm = MPI.COMM_WORLD
  myrank = comm.Get_rank()
  nproc = comm.Get_size()
  N = 1000
  startval = int(N * myrank / nproc + 1)
  endval = int(N * (myrank + 1) / nproc)
  partial_sum = np.array(0, dtype = 'i')

  for i in range (startval, endval + 1):
    partial_sum += i

  if (myrank != 0):
    comm.Send([partial_sum, 1, MPI.INT], dest = 0, tag = 7)

  else:
    tmp_sum = np.array(0, dtype = 'i')
    for i in range (1, nproc):
      comm.Recv([tmp_sum, 1, MPI.INT], source = i, tag = 7)
      partial_sum += tmp_sum

    print('The sum is {0}'.format(partial_sum))