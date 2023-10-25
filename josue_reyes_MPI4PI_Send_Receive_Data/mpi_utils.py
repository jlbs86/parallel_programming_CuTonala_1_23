from mpi4py import MPI

def send_receive_data(data, data_type, source_rank, dest_rank):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == source_rank:
        # Enviar datos desde el origen al destino
        comm.send(data, dest=dest_rank)
    elif rank == dest_rank:
        # Recibir datos en el destino
        received_data = comm.recv(source=source_rank)
        return received_data

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        data_to_send = "Hello, World!"
        received_data = send_receive_data(data_to_send, "STRING", 0, 1)
        print(f"Process {rank} received: {received_data}")
    elif rank == 1:
        data_to_send = [1, 2, 3, 4, 5]
        received_data = send_receive_data(data_to_send, "LIST", 0, 1)
        print(f"Process {rank} received: {received_data}")
