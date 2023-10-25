from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Define a custom function for communication
def custom_communication(rank, data_to_send, dest_rank, source_rank):
    if rank == source_rank:
        # Send data from the source rank to the destination rank
        comm.send(data_to_send, dest=dest_rank)
    elif rank == dest_rank:
        # Receive data sent by the source rank
        received_data = comm.recv(source=source_rank)
        return received_data

# Communication between Rank 0 and Rank 1
if rank == 0:
    data_str = "Hello from Rank 0!"
    custom_communication(rank, data_str, dest_rank=1, source_rank=0)

    data_list = [1, 2, 3, 4, 5]
    received_result = custom_communication(rank, data_list, dest_rank=1, source_rank=1)
    print(f"Rank 0 received result: {received_result}")
elif rank == 1:
    received_data_str = custom_communication(rank, None, dest_rank=0, source_rank=0)
    print(f"Rank 1 received: {received_data_str}")

    received_data_list = custom_communication(rank, None, dest_rank=0, source_rank=1)
    partial_sum = sum(received_data_list)
    custom_communication(rank, partial_sum, dest_rank=0, source_rank=1)
