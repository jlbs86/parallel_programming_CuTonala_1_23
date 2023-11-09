from mpi4py import MPI

class MPIUtils:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def send_data(self, data, dest_rank):
        if self.rank != dest_rank:
            self.comm.send(data, dest=dest_rank)

    def receive_data(self, source_rank):
        if self.rank != source_rank:
            return self.comm.recv(source=source_rank)

if __name__ == "__main__":
    mpi = MPIUtils()
    data = None

    if mpi.rank == 0:
        data = "Hello from Rank 0!"
        mpi.send_data(data, dest_rank=1)
    elif mpi.rank == 1:
        data = mpi.receive_data(source_rank=0)
        print(f"Received data: {data}")