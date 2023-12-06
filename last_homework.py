from mpi4py import MPI


# Define a depth-first search (DFS) function
def dfs(graph, start, goal, visited=None, path=None):
    # Initialize visited and path if necessary
    if visited is None:
        visited = set()
    if path is None:
        path = []

    # Add start to visited and path
    visited.add(start)
    path = path + [start]

    # If start is the goal, return path
    if start == goal:
        return path

    # Iterate through each neighbor of start
    for neighbor in graph[start]:
        # If neighbor is not visited, add to visited and path
        if neighbor not in visited:
            new_path = dfs(graph, neighbor, goal, visited, path)
            # If new path is not None, return new path
            if new_path is not None:
                return new_path

    # If no path is found, return None
    return None

# Define a breadth-first search (BFS) function
def bfs(graph, start, goal):
    # Initialize visited and queue
    visited = set()
    queue = [[start]]

    # Iterate until queue is empty
    while queue:
        # Pop the first path from the queue
        path = queue.pop(0)
        node = path[-1]

        # If node is the goal, return path
        if node == goal:
            return path
        # If node is not visited, add to visited and add each neighbor to queue
        elif node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

    # If no path is found, return None
    return None

# Main function
if __name__ == '__main__':
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Define the graph
    graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['G'],
        'F': ['H'],
        'G': ['H'],
        'H': []
    }

    
    # Define the goal node
    goal = 'H'
    
    # Root process
    if rank == 0:
        # Find paths using DFS and BFS
        dfs_path = dfs(graph, 'A', goal)
        bfs_path = bfs(graph, 'A', goal)
        
        # Send paths to other processes
        for i in range(1, size):
            comm.send(dfs_path, dest=i, tag=1)
            comm.send(bfs_path, dest=i, tag=2)
    
    # Worker processes
    else:
        # Receive paths from root process
        dfs_path = comm.recv(source=0, tag=1)
        bfs_path = comm.recv(source=0, tag=2)
        
        # Process and print paths
        if dfs_path:
            print(f"Process {rank}: Received DFS Path: {dfs_path}")
        if bfs_path:
            print(f"Process {rank}: Received BFS Path: {bfs_path}")
