""""Script for obtain hardware description"""
from mpi4py import MPI

class MpiDanielDiazUtils:
    """class for simplificate mpi4py library use"""

    def __init__(self):
        self.comm = MPI.COMM_WORLD

    def get_rank(self):
        """ 
        return the rank of the process in the communicator
        """
        return self.comm.Get_rank()
    
    def send_and_recv(self, data):
        """
            this method will send data between execution threads
        """
        rank = self.get_rank()

        if rank == 0:
            res = self.comm.recv(source=1)
            print(f"received from rank 1: {res}")
        elif rank == 1:
            self.comm.send(data, dest=0)




if __name__ == "__main__":
    ins = MpiDanielDiazUtils()

    #testing all required data types
    ins.send_and_recv("hola")
    ins.send_and_recv(1)
    ins.send_and_recv(1.0000001)
    ins.send_and_recv([1,2,3])

