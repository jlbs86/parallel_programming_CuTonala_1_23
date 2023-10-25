"""
Class File to Send and Recieve Data Based on MPI Rank
"""

from mpi4py import MPI

class SendReceiveData:
    """
    Class to Manage the Specifics in the Sending of Receiving Data
    Aids in the Defining of Rank and the Managing of Where the Data is Sent
    """
    
    def __init__(self) -> None:
        """
        Initialize the Comunication and Defines Rank and Size

        """
        try:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            print(f'size: {self.size}  rank: {self.rank}')
            return None
        except Exception as generalException:
            print(generalException)
            MPI.Finalize()
            return None
        
    def defineDataToSend(self, data:str) -> str:
        """
        Function to define the data that will be sent
        
        :param data: Variavble containing the data that will be sent between processes
        :type data: str
        :return: String Containing the Data to be Sent
        :rtype: str

        """
        
        if self.rank == 0: return data
        else: return None
        
    def sendData(self) -> bool:
        """
        Function that takes defined data and sends it to the destination process
        
        :return: Boolean value to verify if run correcly
        :rtype: bool
        
        """
        if not self.validateProccessSize(): return False
        
        try:
            if self.rank == 0: 
                self.comm.send(self.defineDataToSend("Hello World!"), dest=1, tag=0)
                return True
        except Exception as generalException:
            print(generalException)
            return False
        
    def recieveData(self) -> bool:
        """
        Funtion that receives data sent and prints it to the console
        
        :return: Boolean value to verify if run correcly
        :rtype: bool

        """
        if not self.validateProccessSize(): return False
        
        try:
            if self.rank == 1:
                received_data = self.comm.recv(source=0, tag=0)
                print(f"Received data: {received_data}")
                return True
        except Exception as generalException:
            print(generalException)
            return False
    
    def validateProccessSize(self) -> bool:
        """
        Function that verifies process size that is saved upon initializing
        
        :return: Boolean value to verify if size is greater than two
        :rtype: bool

        """
        if self.size < 2:
            print("Minimum of two Processes are Required")
            MPI.Finalize()
            return False
        return True
    
    def __delete__(self) -> bool:
        """
        Function to delete the class object and verify that it is deleted        
        
        :return: Boolean value to the object has deleted correcly
        :rtype: bool
        """
        MPI.Finalize()
        
if __name__ == "__main__":
    srd = SendReceiveData()
    if srd.sendData(): print("Data Sent Correcly")
    if srd.recieveData(): print("Data Received Correcly")
    srd.delete()
