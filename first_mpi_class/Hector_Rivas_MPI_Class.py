""" this file is to manage all examples about MPI using the mpi4py librery """

from mpi4py import MPI

"""MPI class"""
class MPI_rank():
    def __init__(self):
        self.mpi = None
    
    """ This method gets the rank """
    def method_for_rank(self):
        rank = MPI.COMM_WORLD.Get_rank()
        print('Rank: ', rank)
        
if __name__ == '__main__':
    instance = MPI_rank()
    instance.method_for_rank()
    