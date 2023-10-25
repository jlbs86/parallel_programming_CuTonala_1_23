""" In this file we work on sending and receiving messages with the mpi4py library """

from mpi4py import MPI

class send_receive_data():
    def __init__(self, arg=None):
        super(send_receive_data, self).__init__()
        self.arg = arg

    """ This method gets the rank as well as send and receive messages"""

    def method_for_rank(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Only continue if there are at least two processes

        if size < 2:
            print("This code requires at least two processes.")
            MPI.Finalize()
            exit()

        # Define the data to be sent
        if rank == 0:
            data_rank0 = "Hello, from process 0!"
        else:
            data_rank0 = None

        # Send data from rank 0 to rank 1
        if rank == 0:
            comm.send(data_rank0, dest=1, tag=0)
            
        # Receive data in rank 1
        if rank == 1:
            received_data0 = comm.recv(source=0, tag=0)
            print("Received data rank 0:", received_data0)
            
        if rank == 1:
            data_rank1 = [1, 2, 3, 4, 5]
        else: 
            data_rank1: None
            
        if rank == 1:
            comm.send(data_rank1, dest=0, tag=1)
            
        if rank == 0:
            received_data1 = comm.recv(source=1, tag=1)
            print("Receive data rank 1:", received_data1)
            

if __name__ == '__main__':
    instance = send_receive_data()
    instance.method_for_rank()
