import csv
import networkx as nx
import matplotlib.pyplot as plt
import time
from Coordinates import Coordinates
from test import Station

g = nx.Graph()  # Generating the graph using networkx

m = nx.Graph()  # Generating maze

with open('Stations.csv', 'r') as csv_file:  # Reading the stations from the CSV file
    csv_reader = csv.reader(csv_file)
    stations = []  # Creating Empty station lists
    lines = []
    u1 = []
    u2 = []
    u3 = []
    u4 = []
    u5 = []
    u6 = []
    u7 = []
    u8 = []
    u9 = []
    u55 = []
    s = ""
    t = 0
    change_stations = []  # Change stations that we can change lines
    change = ""
    station = Station(None, None, None, None, None, None)  # type: Station

    for f in csv_reader:
        station = Station(None, None, None, None, None, None)
        station.set_name(f[0])
        station.set_time(f[1])
        station.get_line().append(f[2])
        station.set_cor_x(f[3])
        station.set_cor_y(f[4])
        stations.append(station)

        s = ""

        s = f[2]
        # change = change + station.get_change()  # Assigning each station according to its line
        if s == "u1":
            u1.append(station)
        if s == "u2":
            u2.append(station)
        if s == "u3":
            u3.append(station)
        if s == "u4":
            u4.append(station)
        if s == "u5":
            u5.append(station)
        if s == "u6":
            u6.append(station)
        if s == "u7":
            u7.append(station)
        if s == "u8":
            u8.append(station)
        if s == "u9":
            u9.append(station)
        if s == "u55":
            u55.append(station)
        s = ""


def inserted_sort(line):  # Sorting the stations based on the time"weight"
    for i in range(1, len(line)):
        for j in range(i - 1, -1, -1):
            if int(line[j].get_time()) > int(line[j + 1].get_time()):
                line[j], line[j + 1] = line[j + 1], line[j]
            else:
                break


inserted_sort(u1)
inserted_sort(u2)
inserted_sort(u3)
inserted_sort(u4)
inserted_sort(u5)
inserted_sort(u55)
inserted_sort(u6)
inserted_sort(u7)
inserted_sort(u8)
inserted_sort(u9)


def generate_graph_cordinates(create_station):  # Generating stations in coordinates
    for j in range(0, len(create_station) - 1):
        w = abs(int(create_station[j].get_time()) - int(create_station[j + 1].get_time()))
        m.add_edge((create_station[j].get_cor_x(), create_station[j].get_cor_y()),
                   (create_station[j + 1].get_cor_x(), create_station[j + 1].get_cor_y()), weight=w)
    for i in range(len(create_station), 0):
        w = abs(int(create_station[i].get_time()) - int(create_station[i - 1].get_time()))
        m.add_edge((create_station[i].get_cor_x(), create_station[i].get_cor_y()),
                   (create_station[i - 1].get_cor_x(), create_station[i - 1].get_cor_y()), weight=w)


def generate_graph(create_station):  # Generating Stations
    for i in range(0, len(create_station) - 1):
        w = abs(int(create_station[i].get_time()) - int(create_station[i + 1].get_time()))
        g.add_edge(create_station[i].get_name(), create_station[i + 1].get_name(), weight=w)

    for i in range(len(create_station), 0):
        w = abs(int(create_station[i].get_time()) - int(create_station[i + 1].get_time()))
        g.add_edge(create_station[i].get_name(), create_station[i - 1].get_name(), weight=w)


generate_graph(u1)
generate_graph(u2)
generate_graph(u3)
generate_graph(u4)
generate_graph(u55)
generate_graph(u5)
generate_graph(u6)
generate_graph(u7)
generate_graph(u8)
generate_graph(u9)

generate_graph_cordinates(u1)
generate_graph_cordinates(u2)
generate_graph_cordinates(u3)
generate_graph_cordinates(u4)
generate_graph_cordinates(u5)
generate_graph_cordinates(u55)
generate_graph_cordinates(u6)
generate_graph_cordinates(u7)
generate_graph_cordinates(u8)
generate_graph_cordinates(u9)


def bfs(g, start,
        goal):  # BFS algorithm, takes parameters of a graph, start station and end station, returns the taken path
    # from start to goal
    queue = [[start]]
    visited = []
    time = 0

    if start == goal:
        return "You're already there!"
    while queue:
        path = queue.pop(0)
        node = path[-1]

        if node not in visited:
            neighbours = list(g.neighbors(node))  # Puts neighbours of "node" in a list

            for neighbour in neighbours:

                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)

                if neighbour == goal:

                    for i in range(0, len(new_path) - 1):
                        time = time + g.get_edge_data(new_path[i], new_path[i + 1])['weight']

                    return "Time Taken = " + str(time) + " mins", new_path
            visited.append(node)

    return "Sorry, trip not possible."


def __findPathDFS(graph, current, goal, visited):  # DFS Algorithm
    if current == goal:
        return [current]

    if current in graph:

        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                path = __findPathDFS(graph, neighbor, goal, visited)

                if path is not None:
                    timeTaken = 0
                    path.insert(0, current)
                    for i in range(0, len(path) - 1):
                        timeTaken = timeTaken + g.get_edge_data(path[i], path[i + 1])['weight']



                    return path
    return None


def findPathDFS(graph, start, goal):  # Recursive method for DFS to get the path
    if start not in graph or goal not in graph:
        return False

    visited = set()
    visited.add(start)

    return __findPathDFS(graph, start, goal, visited)

def get_coordinates(
        station):  # Getting co-ordinates of a station by using its name
    for i in range(0, len(stations)):
        if station == stations[i].get_name():
            return (stations[i].get_cor_x(), stations[i].get_cor_y())


def get_path(station):  # the A* returns co-ordinates of the stations path, this method searches for the station corresponding to the co-ordinates
    path = []
    for i in range(0, len(station)):
        for j in range(0, len(stations)):

            if (station[i]) == ((stations[j].get_cor_x(),stations[j].get_cor_y())):

                if stations[j].get_name() not in path:
                    path.append(stations[j].get_name())
    return path



def TimeTaken(path, graph):  # Time taken between stations
    time = 0
    for i in range(0, len(path) - 1):
        time = time + graph.get_edge_data(path[i], path[i + 1])['weight']
    return time


def heuristic(start, goal):
    # Use Chebyshev distance heuristic if we can move one square either
    # adjacent or diagonal
    D = 1
    D2 = 1
    dx = 0
    dy = 0
    if len(start) > 0 and len(goal) > 0:
        dx = abs(float(start[0]) - float(goal[0]))
        dy = abs(float(start[1]) - float(goal[1]))

    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)


def AStarSearch(start, end, graph):
    G = {}  # Actual movement cost to each position from the start position
    F = {}  # Estimated movement cost of start to end going via this position

    # Initialize starting values
    G[start] = 0
    F[start] = heuristic(start, end)

    closedVertices = set()
    openVertices = set([start])
    cameFrom = {}

    while len(openVertices) > 0:
        # Get the vertex in the open list with the lowest F score
        current = None
        currentFscore = None
        for pos in openVertices:
            if current is None or F[pos] < currentFscore:
                currentFscore = F[pos]
                current = pos

        # Check if we have reached the goal
        if current == end:
            # Retrace our route backward
            path = [current]
            while current in cameFrom:
                current = cameFrom[current]
                path.append(current)
            path.reverse()
            return path, F[end]  # Done!

        # Mark the current vertex as closed
        openVertices.remove(current)
        closedVertices.add(current)

        # Update scores for vertices near the current position

        for neighbour in graph[current]:


            if neighbour in closedVertices:
                continue  # We have already processed this node exhaustively
            candidateG = G[current] + graph.get_edge_data(current, neighbour)['weight']
            if neighbour not in openVertices:
                openVertices.add(neighbour)  # Discovered a new vertex

            elif candidateG >= G[neighbour]:
                continue  # This G score is worse than previously found

            # Adopt this G score
            cameFrom[neighbour] = current
            G[neighbour] = candidateG
            H = heuristic(neighbour, end)
            F[neighbour] = G[neighbour] + H

    raise RuntimeError("A* failed to find a solution")



start = time.time()
print(bfs(g, 'Alexanderplatz', 'Halemweg'))
print('_________________________')
print(findPathDFS(g, 'Alexanderplatz', 'Halemweg'))
print(str(TimeTaken(findPathDFS(g, 'Alexanderplatz', 'Halemweg'), g)) + " mins")
print('______________________________________________')

x = (AStarSearch(get_coordinates('Alexanderplatz'), get_coordinates('Halemweg'), m))[0]
print(get_path(x))
print(str(TimeTaken((AStarSearch(get_coordinates('Alexanderplatz'), get_coordinates('Halemweg'), m))[0], m)) + " mins")
print(start)

nx.draw(g, with_labels=True)

plt.show()
