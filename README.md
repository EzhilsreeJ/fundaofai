<h1>ExpNo 1 : Developing AI Agent with PEAS Description</h1>
<h3>AIM:</h3>
<p>To find the PEAS description for the given AI problem and develop an AI agent.</p>
<h3>Theory</h3>

<h3>Room-Cleaning Agent:</h3>
<p>
An <strong>AI agent</strong> is a system capable of autonomous actions in an environment to achieve a goal. It senses through <strong>sensors</strong> and acts through <strong>actuators</strong>.
The agent's behavior can be modeled using the <strong>PEAS</strong> framework:
</p>

<ul>
  <li><strong>P</strong> – Performance Measure</li>
  <li><strong>E</strong> – Environment</li>
  <li><strong>A</strong> – Actuators</li>
  <li><strong>S</strong> – Sensors</li>
</ul>

<p>
In this experiment, we simulate a <strong>room-cleaning agent</strong> (like a vacuum cleaner), operating in <strong>two rooms (A and B)</strong>. The agent must:
</p>

<ul>
  <li>Check for dirt in the current room.</li>
  <li>Clean if dirty.</li>
  <li>Move to the next room if needed.</li>
</ul>

<p>
Each <strong>cleaning</strong> action improves the performance, while each <strong>movement</strong> reduces it.
</p>

---

<h3>PEAS DESCRIPTION:</h3>

<table>
  <tr>
    <th>Agent Type</th>
    <th>Performance Measure</th>
    <th>Environment</th>
    <th>Actuators</th>
    <th>Sensors</th>
  </tr>
  <tr>
    <td>Room-Cleaning Agent</td>
    <td>Cleanliness, Fewer moves</td>
    <td>Rooms A and B</td>
    <td>Move Left/Right, Clean</td>
    <td>Location Sensor, Dirt Sensor</td>
  </tr>
</table>


<h3>DESIGN STEPS</h3>

<h4>STEP 1: Identifying the input:</h4>
<p>Location of the agent, Dirt status of the rooms</p>

<h4>STEP 2: Identifying the output:</h4>
<p>Actions: Move, Clean, Do nothing</p>

<h4>STEP 3: Developing the PEAS description:</h4>
<p>PEAS description is defined by performance, environment, actuators, and sensors in the AI agent model.</p>

<h4>STEP 4: Implementing the AI agent:</h4>
<p>Simulate the agent behavior to clean dirty rooms and move as required.</p>

<h4>STEP 5: Measure the performance:</h4>
<p>Increment for cleaning action, decrement for each movement.</p>


<h3>PROGRAM:</h3>

```python
# Developing AI Agent with PEAS Description
# Developed by: EZHIL SREE J
# Register Number: 212223230056

class VacuumCleanerAgent:
    def __init__(self):
        # Initialize the agent's state (location and dirt status)
        self.location = "A"  # Initial location (can be "A" or "B")
        self.dirt_status = {"A": False, "B": False}  # Initial dirt status (False means no dirt)

    def move_left(self):
        # Move the agent to the left if possible
        if self.location == "B":
            self.location = "A"

    def move_right(self):
        # Move the agent to the right if possible
        if self.location == "A":
            self.location = "B"

    def suck_dirt(self):
        # Suck dirt in the current location if there is dirt
        if self.dirt_status[self.location]:
            self.dirt_status[self.location] = False
            print(f"Sucked dirt in location {self.location}")

    def do_nothing(self):
        # Do nothing
        pass

    def perform_action(self, action):
        # Perform the specified action
        if action == "left":
            self.move_left()
        elif action == "right":
            self.move_right()
        elif action == "suck":
            self.suck_dirt()
        elif action == "nothing":
            self.do_nothing()
        else:
            print("Invalid action")

    def print_status(self):
        # Print the current status of the agent
        print(f"Location: {self.location}, Dirt Status: {self.dirt_status}")

# Example usage:
agent = VacuumCleanerAgent()

# Move the agent, suck dirt, and do nothing
agent.perform_action("left")
agent.print_status()
agent.perform_action("suck")
agent.print_status()
agent.perform_action("nothing")
agent.print_status()
```
## OUTPUT:
![Screenshot 2025-05-19 205304](https://github.com/user-attachments/assets/49506727-fdbe-4ef2-8b45-b20b91bf4293)

## RESULT:
Thus the Developing AI Agent with PEAS Description was implemented using python programming.
---------------------------------------------------------------------------------------------------------------------------

# ExpNo 2 : Implement Depth First Search Traversal of a Graph
# Aim:
To Implement Depth First Search Traversal of a Graph using Python 3.
# Algorithm:
1.Construct a Graph with Nodes and Edges
2.Depth First Search Uses Stack and Recursion
3.Insert a START node to the STACK
4.Find its Successors Or neighbors and Check whether the node is visited or not
5.If Not Visited, add it to the STACK. Else Call The Function Again Until No more nodes needs to be visited.
# program
```
from collections import defaultdict

def dfs(graph, start, visited, path):
    path.append(start)
    visited[start] = True
    for neighbour in graph[start]:
        if not visited[neighbour]:
            dfs(graph, neighbour,
                visited, path)
    return path

graph = defaultdict(list)
n, e = map(int, input().split())

for i in range(e):
    u, v = input().split()
    graph[u].append(v)
    graph[v].append(u) 

if '0' in graph:
    start = '0'
else:
    start = 'A'
visited = defaultdict(bool)
path = []

traversed_path = dfs(graph, start, visited, path)
print(traversed_path)
```
# Sample Input
8 9
A B
A C
B E
C D
B D
C G
D F
G F
F H
# Sample Output
['A', 'B', 'E', 'D', 'C', 'G', 'F', 'H']
# Sample Input
5 5
0 1
0 2
0 3
2 3
2 4
# Sample Output
['0', '1', '2', '3', '4']
# Result:
Thus,a Graph was constructed and implementation of Depth First Search for the same graph was done successfully.
--------------------------------------------------------------------------------------------------------------------
# ExpNo 3 : Implement Breadth First Search Traversal of a Graph
# Aim:
To Implement Breadth First Search Traversal of a Graph using Python 3.
# Algorithm:
1.Construct a Graph with Nodes and Edges
2.Breadth First Uses Queue and iterates through the Queue for Traversal.
3.Insert a Start Node into the Queue.
4.Find its Successors Or neighbors and Check whether the node is visited or not.
5.If Not Visited, add it to the Queue. Else Continue.
6.Iterate steps 4 and 5 until all nodes get visited, and there are no more unvisited nodes.

# program
```
from collections import deque
from collections import defaultdict


'''
V E
FOR EVERY EDGE
U V
7 9
A B
A C 
A F
C E
C F
C D
D E 
D G
G F
'''
def bfs(graph,start,visited,path):
    queue = deque()
    path.append(start)
    queue.append(start)
    visited[start] = True
    while len(queue) != 0:
        tmpnode = queue.popleft()
        for neighbour in graph[tmpnode]:
            if visited[neighbour] == False:
                path.append(neighbour)
                queue.append(neighbour)
                visited[neighbour] = True
    return path

graph = defaultdict(list)
v,e = map(int,input().split())
for i in range(e):
    u,v = map(str,input().split())
    graph[u].append(v)
    graph[v].append(u)

if '0' in graph:
    start = '0'
else:
    start = 'A'
path = []
visited = defaultdict(bool)
traversedpath = bfs(graph,start,visited,path)
print(traversedpath)
```
# Sample Input
7 9
A B
A C
A F
C E
C F
C D
D E
D G
G F
# Sample Output
['A', 'B', 'C', 'F', 'E', 'D', 'G']
# Sample Input
5 6
0 1
0 2
1 2
1 3
2 4
3 4
# Sample Output
['0', '1', '2', '3', '4']
# Result:
Thus,a Graph was constructed and implementation of Breadth First Search for the same graph was done successfully.
----------------------------------------------------------------------------------------------------------
# ExpNo 4 : Implement A* search algorithm for a Graph

# Aim:
To ImplementA * Search algorithm for a Graph using Python 3.
# Algorithm:
// A* Search Algorithm

1.Initialize the open list

2.Initialize the closed list put the starting node on the open list (you can leave its f at zero)

3.while the open list is not empty 

a) find the node with the least f on the open list, call it "q"

b) pop q off the open list

c) generate q's 8 successors and set their parents to q

d) for each successor

i) if successor is the goal, stop search

ii) else, compute both g and h for successor successor.g = q.g + distance between successor and q successor.h = distance from goal to successor (This can be done using many ways, we will discuss three heuristics- Manhattan, Diagonal and Euclidean Heuristics) successor.f = successor.g + successor.h

iii) if a node with the same position as successor is in the OPEN list which has a lower f than successor, skip this successor

iv) if a node with the same position as successor is in the CLOSED list which has a lower f than successor, skip this successor otherwise, add the node to the open list end (for loop)

e) push q on the closed list end (while loop)

# program
```
from collections import defaultdict

def heuristic(node, H_dist):
    return H_dist.get(node, 0)

def get_neighbors(node, Graph_nodes):
    return Graph_nodes.get(node, [])

def aStarAlgo(start_node, stop_node, Graph_nodes, H_dist):
    open_set = set([start_node])
    closed_set = set()
    g = {}  # store distance from starting node
    parents = {}  # parents contains an adjacency map of all nodes

    # distance of starting node from itself is zero
    g[start_node] = 0

    # start_node is root node i.e it has no parent nodes
    # so start_node is set to its own parent node
    parents[start_node] = start_node

    while len(open_set) > 0:
        n = None

        # node with lowest f() is found
        for v in open_set:
            if n is None or g[v] + heuristic(v, H_dist) < g[n] + heuristic(n, H_dist):
                n = v

        if n is None:
            print('Path does not exist!')
            return None

        # if the current node is the stop_node
        # then we begin reconstructing the path from it to the start_node
        if n == stop_node:
            path = []
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            path.append(start_node)
            path.reverse()
            print('Path found: {}'.format(path))
            return path

        # for all the neighbors of the current node do
        for (m, weight) in get_neighbors(n, Graph_nodes):
            # nodes 'm' not in first and last set are added to first
            # n is set its parent
            if m not in open_set and m not in closed_set:
                open_set.add(m)
                parents[m] = n
                g[m] = g[n] + weight
            # for each node m, compare its distance from start i.e g(m) to the
            # from start through n node
            else:
                if g[m] > g[n] + weight:
                    # update g(m)
                    g[m] = g[n] + weight
                    # change parent of m to n
                    parents[m] = n
                    # if m in closed set, remove and add to open
                    if m in closed_set:
                        closed_set.remove(m)
                        open_set.add(m)

        # remove n from the open_list, and add it to closed_list
        # because all of its neighbors were inspected
        open_set.remove(n)
        closed_set.add(n)

    print('Path does not exist!')
    return None

# Input from user
Graph_nodes = defaultdict(list)
num_edges = int(input("Enter the number of edges: "))
print("Enter the edges in the format 'start end weight':")
for _ in range(num_edges):
    start, end, weight = input().split()
    weight = int(weight)
    Graph_nodes[start].append((end, weight))
    Graph_nodes[end].append((start, weight))  # if the graph is undirected

H_dist = {}
num_nodes = int(input("Enter the number of nodes: "))
print("Enter the heuristic values in the format 'node heuristic_value':")
for _ in range(num_nodes):
    node, h_value = input().split()
    H_dist[node] = int(h_value)

start_node = input("Enter the start node: ")
stop_node = input("Enter the stop node: ")

aStarAlgo(start_node, stop_node, Graph_nodes, H_dist)
```
# sample graph I
![image](https://github.com/natsaravanan/19AI405FUNDAMENTALSOFARTIFICIALINTELLIGENCE/assets/87870499/b1377c3f-011a-4c0f-a843-516842ae056a)

# sample input
10 14
A B 6
A F 3
B D 2
B C 3
C D 1
C E 5
D E 8
E I 5
E J 5
F G 1
G I 3
I J 3
F H 7
I H 2
A 10
B 8
C 5
D 7
E 3
F 6
G 5
H 3
I 1
J 0
# Sample Output
Path found: ['A', 'F', 'G', 'I', 'J']

# sample graph II
![image](https://github.com/natsaravanan/19AI405FUNDAMENTALSOFARTIFICIALINTELLIGENCE/assets/87870499/acbb09cb-ed39-48e5-a59b-2f8d61b978a3)
# Sample Input
6 6
A B 2
B C 1
A E 3
B G 9
E D 6
D G 1
A 11
B 6
C 99
E 7
D 1
G 0
# Sample Output
Path found: ['A', 'E', 'D', 'G']
# Result:
Implementing A * Search algorithm for a Graph using Python 3. is executed successfully.
----------------------------------------------------------------------------------------------------
# ExpNo 5 : Implement Simple Hill Climbing Algorithm
# Aim:
Implement Simple Hill Climbing Algorithm and Generate a String by Mutating a Single Character at each iteration
# Algorithm:

1.Evaluate the initial state.If it is a goal state then return it and quit. Otherwise, continue with initial state as current state.
2.Loop until a solution is found or there are no new operators left to be applied in current state:
(i)Select an operator that has not yet been applied to the current state and apply it to produce a new state
(ii)Evaluate the new state:
     (a)if it is a goal state, then return it and quit
     (b)if it is not a goal state but better than current state then make new state as current state
     (c)if it is not better than current state then continue in the loop
# Steps Applied:
Step-1
Generate Random String of the length equal to the given String

Step-2
Mutate the randomized string each character at a time

Step-3
Evaluate the fitness function or Heuristic Function

Step-4:
Lopp Step -2 and Step-3 until we achieve the score to be Zero to achieve Global Minima.

# program
```
import random
import string

def generate_random_solution(answer):
    l = len(answer) 
    return [random.choice(string.printable) for _ in range(l)]

def evaluate(solution, answer):
    print("Solution:", "".join(solution))
    target = list(answer)
    diff = 0
    for i in range(len(target)):
        s = solution[i]
        t = target[i]
        diff += abs(ord(s) - ord(t))
    return diff

def mutate_solution(solution):
    ind = random.randint(0, len(solution) - 1)
    solution[ind] = random.choice(string.printable)
    return solution

def SimpleHillClimbing():
    answer = "Artificial Intelligence"
    best = generate_random_solution(answer)
    best_score = evaluate(best, answer)
    while True:
        print("Score:", best_score, "Solution:", "".join(best))
        if best_score == 0:
            break
        new_solution = mutate_solution(list(best))
        score = evaluate(new_solution, answer)
        if score < best_score:
            best = new_solution
            best_score = score


# answer = "Artificial Intelligence"
# print(generate_random_solution(answer))
# solution = generate_random_solution(answer)
# print(evaluate(solution, answer))

SimpleHillClimbing()
```
# Sample Input and Output
# Sample String:
Artificial Intelligence

# Output:
Score: 643 Solution : 8RzF:oG ]%;CPORRMe!zGvk
Score: 609 Solution : 8RzF:oG ]%;CPqRRMe!zGvk
Score: 604 Solution : 8RzF:oG ]%;CPqRRMe!zGqk
Score: 594 Solution : 8RzF:oG ]%;CPqRRWe!zGqk
Score: 551 Solution : 8RzF:oGK]%;CPqRRWe!zGqk
Score: 551 Solution : 8RzF:oGK]%;CPqRRWe!zGqk
Score: 551 Solution : 8RzF:oGK]%;CPqRRWe!zGqk
Score: 551 Solution : 8RzF:oGK]%;CPqRRWe!zGqk
Score: 551 Solution : 8RzF:oGK]%;CPqRRWe!zGqk
....................................................
..................................................
................................................
Score: 1 Solution : Artificial Intelligencf
Score: 1 Solution : Artificial Intelligencf
Score: 1 Solution : Artificial Intelligencf
Score: 1 Solution : Artificial Intelligencf
Score: 0 Solution : Artificial Intelligence
# RESULT:
The Simple Hill Climbing algorithm successfully generated the target string "Artificial Intelligence" by iteratively mutating and improving the solution based on heuristic evaluation.
--------------------------------------------------------------------------------------
# ExpNo 6 : Implement Minimax Search Algorithm for a Simple TIC-TAC-TOE game

# Aim:
Implement Minimax Search Algorithm for a Simple TIC-TAC-TOE game

# algorithm
1.Initialize a 3×3 board and set player X to start the game.

2.Repeat: display the board and check if any player has won or if it's a tie.

3.If it is X’s turn, get a valid move from the user; otherwise compute O’s best move using max().

4.In max(), try all empty cells by placing O, call min(), and choose the move with the highest score.

5.In min(), try all empty cells by placing X, call max(), and choose the move with the lowest score.

6.Terminal states return +1 (O wins), –1 (X wins), or 0 (tie) to the Minimax functions.

7.Update the board with the selected move and switch the player.

8.Continue until a winner is found or the board becomes full.

# program
```
import time

class Game:
    def __init__(self):
        self.initialize_game()

    def initialize_game(self):
        self.current_state = [['.','.','.'],
                              ['.','.','.'],
                              ['.','.','.']]
        self.player_turn = 'X'  # Player X always plays first

    def draw_board(self):
        for i in range(3):
            for j in range(3):
                print('{}|'.format(self.current_state[i][j]), end=" ")
            print()
        print()

    def is_valid(self, px, py):
        if px < 0 or px > 2 or py < 0 or py > 2:
            return False
        elif self.current_state[px][py] != '.':
            return False
        else:
            return True

    def is_end(self):
        # Vertical win
        for i in range(3):
            if (self.current_state[0][i] != '.' and
                self.current_state[0][i] == self.current_state[1][i] and
                self.current_state[1][i] == self.current_state[2][i]):
                return self.current_state[0][i]

        # Horizontal win
        for i in range(3):
            if (self.current_state[i] == ['X', 'X', 'X']):
                return 'X'
            elif (self.current_state[i] == ['O', 'O', 'O']):
                return 'O'

        # Main diagonal win
        if (self.current_state[0][0] != '.' and
            self.current_state[0][0] == self.current_state[1][1] and
            self.current_state[0][0] == self.current_state[2][2]):
            return self.current_state[0][0]

        # Second diagonal win
        if (self.current_state[0][2] != '.' and
            self.current_state[0][2] == self.current_state[1][1] and
            self.current_state[0][2] == self.current_state[2][0]):
            return self.current_state[0][2]

        # Check for a tie
        for i in range(3):
            for j in range(3):
                if self.current_state[i][j] == '.':
                    return None  # Game is still ongoing

        return '.'  # It's a tie

    def max(self):
        maxv = -2  # Initial worse case
        px = None
        py = None

        result = self.is_end()
        if result == 'X':
            return (-1, 0, 0)
        elif result == 'O':
            return (1, 0, 0)
        elif result == '.':
            return (0, 0, 0)

        for i in range(3):
            for j in range(3):
                if self.current_state[i][j] == '.':
                    self.current_state[i][j] = 'O'  # Simulate AI's move
                    (m, min_i, min_j) = self.min()  # Call min() to evaluate
                    if m > maxv:
                        maxv = m
                        px = i
                        py = j
                    self.current_state[i][j] = '.'  # Undo the move

        return (maxv, px, py)

    def min(self):
        minv = 2  # Initial worse case
        qx = None
        qy = None

        result = self.is_end()
        if result == 'X':
            return (-1, 0, 0)
        elif result == 'O':
            return (1, 0, 0)
        elif result == '.':
            return (0, 0, 0)

        for i in range(3):
            for j in range(3):
                if self.current_state[i][j] == '.':
                    self.current_state[i][j] = 'X'  # Simulate player's move
                    (m, max_i, max_j) = self.max()  # Call max() to evaluate
                    if m < minv:
                        minv = m
                        qx = i
                        qy = j
                    self.current_state[i][j] = '.'  # Undo the move

        return (minv, qx, qy)

    def play(self):
        while True:
            self.draw_board()
            self.result = self.is_end()

            if self.result != None:
                if self.result == 'X':
                    print('The winner is X!')
                elif self.result == 'O':
                    print('The winner is O!')
                elif self.result == '.':
                    print("It's a tie!")

                self.initialize_game()
                return

            if self.player_turn == 'X':
                while True:
                    start = time.time()
                    (m, qx, qy) = self.min()
                    end = time.time()
                    print('Evaluation time: {}s'.format(round(end - start, 7)))
                    print('Recommended move: X = {}, Y = {}'.format(qx, qy))

                    px = int(input('Insert the X coordinate: '))
                    py = int(input('Insert the Y coordinate: '))

                    if self.is_valid(px, py):
                        self.current_state[px][py] = 'X'
                        self.player_turn = 'O'
                        break
                    else:
                        print('The move is not valid! Try again.')

            else:
                (m, px, py) = self.max()
                self.current_state[px][py] = 'O'
                self.player_turn = 'X'

def main():
    g = Game()
    g.play()

if __name__ == "__main__":
    main()
```
# sample input and output
![image](https://github.com/natsaravanan/19AI405FUNDAMENTALSOFARTIFICIALINTELLIGENCE/assets/87870499/6b668685-8bcc-43c5-b5c2-ddd43f3da84a)
![image](https://github.com/natsaravanan/19AI405FUNDAMENTALSOFARTIFICIALINTELLIGENCE/assets/87870499/8ca1b08a-8312-4ef5-89df-e69b7b2c3fa2)
![image](https://github.com/natsaravanan/19AI405FUNDAMENTALSOFARTIFICIALINTELLIGENCE/assets/87870499/dc06427a-d4ce-43a1-95bd-9acfaefac323)
![image](https://github.com/natsaravanan/19AI405FUNDAMENTALSOFARTIFICIALINTELLIGENCE/assets/87870499/a8a27e2a-6fd4-46a2-afb5-6d27b8556702)
![image](https://github.com/natsaravanan/19AI405FUNDAMENTALSOFARTIFICIALINTELLIGENCE/assets/87870499/a2acb6a1-ed8e-42e5-8968-fe805e4b0255)
# Result:
Thus,Implementation of Minimax Search Algorithm for a Simple TIC-TAC-TOE game wasa done successfully.
