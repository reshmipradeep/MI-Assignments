"""
You can create any other helper funtions.
Do not modify the given functions
"""
import queue
import copy

def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """

    path = []
    explored = set()
    frontier = queue.PriorityQueue()
    frontier.put((heuristic[start_point],(start_point, [start_point], 0)))
    length = len(cost) 

    while frontier:
        details= frontier.get()
        tuple = details[1]
        node = tuple[0]
        path = tuple[1]
        nodecost = tuple[2]

        if node not in explored:
            explored.add(node)

            if node in goals:
                    return path

            for neighbor in range(length-1, 0, -1): #lexicographically
                if neighbor not in explored and cost[node][neighbor] > 0:
                    newpath=copy.deepcopy(path)
                    newpath.append(neighbor)

                    newCost = nodecost + cost[node][neighbor] #cost till neighbor node
                    newTotalCost = newCost + heuristic[neighbor]

                    frontier.put((newTotalCost,(neighbor,newpath,newCost)))

    return list() #if goal not found         

def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    path = []
    explored = set()
    frontier = queue.LifoQueue() 
    frontier.put((start_point, [start_point]))
    length = len(cost)    

    while frontier:
        tuple = frontier.get()
        node=tuple[0]
        path=tuple[1]

        if node not in explored:
            explored.add(node)
        
            if node in goals:
                return path

            for neighbor in range(length-1, 0, -1): #lexicographically
                if neighbor not in explored and cost[node][neighbor] > 0:
                    newpath=copy.deepcopy(path)
                    newpath.append(neighbor)
                    frontier.put((neighbor,newpath))

    return list() #empty if goal not found
