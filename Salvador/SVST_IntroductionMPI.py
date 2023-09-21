"""SVST_IntroductionMPI"""

import mpi4py as MPI
import numpy as np
import sys

class Introduction_Mpi:
    """Introduction_Mpi
        Class that contains a few basic implementations of the mpi4py library
    """
    def point_to_point1() -> bool:    
        """
        Python objects (pickle under the hood)
        
        :return: Boolean value to verify if run correcly
        :rtype: bool

        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            data = {'a': 7, 'b': 3.14}
            comm.send(data, dest=1, tag=11)
        elif rank == 1:
            data = comm.recv(source=0, tag=11)
        return True
        
    def point_to_point2() -> bool:
        """
        Python objects with non-blocking communication
        
        :return: Boolean value to verify if run correcly
        :rtype: bool

        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            data = {'a': 7, 'b': 3.14}
            req = comm.isend(data, dest=1, tag=11)
            req.wait()
        elif rank == 1:
            req = comm.irecv(source=0, tag=11)
            data = req.wait()
        return True
    
    def point_to_point3() -> bool:
        """
        NumPy arrays (the fast way!)
        
        :return: Boolean value to verify if run correcly
        :rtype: bool

        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        # passing MPI datatypes explicitly
        if rank == 0:
            data = np.arange(1000, dtype='i')
            comm.Send([data, MPI.INT], dest=1, tag=77)
        elif rank == 1:
            data = np.empty(1000, dtype='i')
            comm.Recv([data, MPI.INT], source=0, tag=77)
        
        # automatic MPI datatype discovery
        if rank == 0:
            data = np.arange(100, dtype=np.float64)
            comm.Send(data, dest=1, tag=13)
        elif rank == 1:
            data = np.empty(100, dtype=np.float64)
            comm.Recv(data, source=0, tag=13)
        return True
    
    def collective_Communication1() -> bool:

        """
        Broadcasting a Python dictionary
        
        :return: Boolean value to verify if run correcly
        :rtype: bool

        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            data = {'key1' : [7, 2.72, 2+3j],
                    'key2' : ( 'abc', 'xyz')}
        else:
            data = None
        data = comm.bcast(data, root=0)
        return True
    
    def collective_Communication2() -> bool:
        """
        Scattering Python objects
        
        :return: Boolean value to verify if run correcly
        :rtype: bool

        """
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        
        if rank == 0:
            data = [(i+1)**2 for i in range(size)]
        else:
            data = None
        data = comm.scatter(data, root=0)
        assert data == (rank+1)**2
        return True
    
    def collective_Communication3() -> bool:
        """
        Gathering Python objects
        
        :return: Boolean value to verify if run correcly
        :rtype: bool

        """
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
        return True
    
    def collective_Communication4() -> bool:
        """
        Broadcasting a NumPy array:
        
        :return: Boolean value to verify if run correcly
        :rtype: bool

        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            data = np.arange(100, dtype='i')
        else:
            data = np.empty(100, dtype='i')
        comm.Bcast(data, root=0)
        for i in range(100):
            assert data[i] == i
        return True
    
    def collective_Communication5() -> bool:
        """
        Scattering NumPy arrays
        
        :return: Boolean value to verify if run correcly
        :rtype: bool

        """
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
        return True
    
    def collective_Communication6() -> bool:
        """
        Gathering NumPy arrays
        
        :return: Boolean value to verify if run correcly
        :rtype: bool

        """
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
        return True

    def mpi_io() -> bool:
        """
        Collective I/O with NumPy arrays
        
        :return: Boolean value to verify if run correcly
        :rtype: bool

        """
        amode = MPI.MODE_WRONLY|MPI.MODE_CREATE
        comm = MPI.COMM_WORLD
        fh = MPI.File.Open(comm, "./datafile.contig", amode)
        
        buffer = np.empty(10, dtype=np.int)
        buffer[:] = comm.Get_rank()
        
        offset = comm.Get_rank()*buffer.nbytes
        fh.Write_at_all(offset, buffer)
        
        fh.Close()
    
    def mpi_io2() -> bool:
        """
        Non-contiguous Collective I/O with NumPy arrays and datatypes
        
        :return: Boolean value to verify if run correcly
        :rtype: bool

        """
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
        return True
    
    def dynamic_proccess_management() -> bool:
        """
        Compute Pi - Master (or parent, or client) side:
        
        :return: Boolean value to verify if run correcly
        :rtype: bool

        """
        comm = MPI.COMM_SELF.Spawn(sys.executable, args=['cpi.py'],maxprocs=5)
        N = np.array(100, 'i')
        comm.Bcast([N, MPI.INT], root=MPI.ROOT)
        PI = np.array(0.0, 'd')
        comm.Reduce(None, [PI, MPI.DOUBLE],
                    op=MPI.SUM, root=MPI.ROOT)
        print(PI)
        comm.Disconnect()
        return True
    
    def dynamic_proccess_management2() -> bool:
        """
        Compute Pi - Worker (or child, or server) side:
        
        :return: Boolean value to verify if run correcly
        :rtype: bool

        """
        comm = MPI.Comm.Get_parent()
        size = comm.Get_size()
        rank = comm.Get_rank()
        
        N = np.array(0, dtype='i')
        comm.Bcast([N, MPI.INT], root=0)
        h = 1.0 / N; s = 0.0
        for i in range(rank, N, size):
            x = h * (i + 0.5)
            s += 4.0 / (1.0 + x**2)
        PI = np.array(s * h, dtype='d')
        comm.Reduce([PI, MPI.DOUBLE], None,
                    op=MPI.SUM, root=0)
        
        comm.Disconnect()
        return True
        
