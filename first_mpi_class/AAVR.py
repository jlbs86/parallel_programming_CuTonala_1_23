from mpi4py import MPI
import numpy as np
import sys

class MPI_AAVR(object):
	"""docstring for MPI_AAVR"""
	def __init__(self, arg):
		super(MPI_AAVR, self).__init__()
		self.arg = arg
		

	    data = None
	def communication_PointToPoint():
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()

		if rank == 0:
		    data = {'a': 7, 'b': 3.14}
		    comm.send(data, dest=1, tag=11)
		elif rank == 1:
		    data = comm.recv(source=0, tag=11)

	def communication_no_blocks():
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()

		if rank == 0:
		    data = {'a': 7, 'b': 3.14}
		    req = comm.isend(data, dest=1, tag=11)
		    req.wait()
		elif rank == 1:
		    req = comm.irecv(source=0, tag=11)
		    data = req.wait()

	def matrix_numPY():
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()

		# passing MPI datatypes explicitly
		if rank == 0:
		    data = numpy.arange(1000, dtype='i')
		    comm.Send([data, MPI.INT], dest=1, tag=77)
		elif rank == 1:
		    data = numpy.empty(1000, dtype='i')
		    comm.Recv([data, MPI.INT], source=0, tag=77)

		# automatic MPI datatype discovery
		if rank == 0:
		    data = numpy.arange(100, dtype=numpy.float64)
		    comm.Send(data, dest=1, tag=13)
		elif rank == 1:
		    data = numpy.empty(100, dtype=numpy.float64)
		    comm.Recv(data, source=0, tag=13)	

	""" Collective communication"""

	def diffusion_dictionary():
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()

		if rank == 0:
		    data = {'key1' : [7, 2.72, 2+3j],
		            'key2' : ( 'abc', 'xyz')}
		else:
		data = comm.bcast(data, root=0)

	def dispersion_object():
		comm = MPI.COMM_WORLD
		size = comm.Get_size()
		rank = comm.Get_rank()

		if rank == 0:
		    data = [(i+1)**2 for i in range(size)]
		else:
		    data = None
		data = comm.scatter(data, root=0)
		assert data == (rank+1)**2

	def collection_object():
		comm = MPI.COMM_WORLD
		size = comm.Get_size()
		rank = comm.Get_rank()

		data = (rank+1)**2
		data = comm.gather(data, root=0)
		if rank == 0:
		    for i in range(size):
		        assert data[i] == (i+1)**2
		else:
		    assert data is None

	def matrix_diffusion():
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()

		if rank == 0:
		    data = np.arange(100, dtype='i')
		else:
		    data = np.empty(100, dtype='i')
		comm.Bcast(data, root=0)
		for i in range(100):
		    assert data[i] == i

	def matrix_dispersion():
		comm = MPI.COMM_WORLD
		size = comm.Get_size()
		rank = comm.Get_rank()

		sendbuf = None
		if rank == 0:
		    sendbuf = np.empty([size, 100], dtype='i')
		    sendbuf.T[:,:] = range(size)
		recvbuf = np.empty(100, dtype='i')
		comm.Scatter(sendbuf, recvbuf, root=0)
		assert np.allclose(recvbuf, rank)

	def matrix_collection():
		comm = MPI.COMM_WORLD
		size = comm.Get_size()
		rank = comm.Get_rank()

		sendbuf = np.zeros(100, dtype='i') + rank
		recvbuf = None
		if rank == 0:
		    recvbuf = np.empty([size, 100], dtype='i')
		comm.Gather(sendbuf, recvbuf, root=0)
		if rank == 0:
		    for i in range(size):
		        assert np.allclose(recvbuf[i,:], i)

	def matvec(comm, A, x):
	    m = A.shape[0] # local rows
	    p = comm.Get_size()
	    xg = numpy.zeros(m*p, dtype='d')
	    comm.Allgather([x,  MPI.DOUBLE],
	                   [xg, MPI.DOUBLE])
	    y = numpy.dot(A, xg)
	    return y

	def matrix_NumPY_collective():
		amode = MPI.MODE_WRONLY|MPI.MODE_CREATE
		comm = MPI.COMM_WORLD
		fh = MPI.File.Open(comm, "./datafile.contig", amode)

		buffer = np.empty(10, dtype=np.int)
		buffer[:] = comm.Get_rank()

		offset = comm.Get_rank()*buffer.nbytes
		fh.Write_at_all(offset, buffer)

		fh.Close()

	def matrix_NumPY_no_contiguous():
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()
		size = comm.Get_size()

		amode = MPI.MODE_WRONLY|MPI.MODE_CREATE
		fh = MPI.File.Open(comm, "./datafile.noncontig", amode)

		item_count = 10

		buffer = np.empty(item_count, dtype='i')
		buffer[:] = rank

		filetype = MPI.INT.Create_vector(item_count, 1, size)
		filetype.Commit()

		displacement = MPI.INT.Get_size()*rank
		fh.Set_view(displacement, filetype=filetype)

		fh.Write_all(buffer)
		filetype.Free()
		fh.Close()

	def proccess_client():
		comm = MPI.COMM_SELF.Spawn(sys.executable,
		                           args=['cpi.py'],
		                           maxprocs=5)

		N = numpy.array(100, 'i')
		comm.Bcast([N, MPI.INT], root=MPI.ROOT)
		PI = numpy.array(0.0, 'd')
		comm.Reduce(None, [PI, MPI.DOUBLE],
		            op=MPI.SUM, root=MPI.ROOT)
		print(PI)

		comm.Disconnect()

	def proccess_worker():
		comm = MPI.Comm.Get_parent()
		size = comm.Get_size()
		rank = comm.Get_rank()

		N = numpy.array(0, dtype='i')
		comm.Bcast([N, MPI.INT], root=0)
		h = 1.0 / N; s = 0.0
		for i in range(rank, N, size):
		    x = h * (i + 0.5)
		    s += 4.0 / (1.0 + x**2)
		PI = numpy.array(s * h, dtype='d')
		comm.Reduce([PI, MPI.DOUBLE], None,
		            op=MPI.SUM, root=0)

		comm.Disconnect()