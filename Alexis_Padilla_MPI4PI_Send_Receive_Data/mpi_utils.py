from mpi4py import MPI

class DataCommunicator:
    def __init__(self, arg=None):
        super(DataCommunicator, self).__init__()
        self.arg = arg

    def perform_data_exchange(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if size < 2:
            print("This code requires at least two processes.")
            MPI.Finalize()
            exit()

        if rank == 0:
            data_rank0 = "Hello, from process 0!"
        else:
            data_rank0 = None

        if rank == 0:
            comm.send(data_rank0, dest=1, tag=0)

        if rank == 1:
            received_data0 = comm.recv(source=0, tag=0)
            print("Received data rank 0:", received_data0)

        if rank == 1:
            data_rank1 = [1, 2, 3, 4, 5]
        else: 
            data_rank1 = None

        if rank == 1:
            comm.send(data_rank1, dest=0, tag=1)

        if rank == 0:
            received_data1 = comm.recv(source=1, tag=1)
            print("Receive data rank 1:", received_data1)

if __name__ == '__main__':
    data_processor = DataCommunicator()
    data_processor.perform_data_exchange()