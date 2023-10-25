from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
  data_to_send = "Hello from Rank 0!"
  comm.send(data_to_send, dest=1)
elif rank == 1:
  received_data = comm.recv(source=0)
  print(f"Rank 1 received: {received_data}")

if rank == 0:
    data_to_send = [1, 2, 3, 4, 5]
    comm.send(data_to_send, dest=1)
    received_result = comm.recv(source=1)
    print(f"Rank 0 received result: {received_result}")
elif rank == 1:
    received_data = comm.recv(source=0)
    partial_sum = sum(received_data)
    comm.send(partial_sum, dest=0)