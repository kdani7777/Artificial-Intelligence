import heapq

"""
Gives us the manhattan distance from each tile to the goal
"""
def calc_manhattan_distance(state):
    distance = 0
    # goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    goal_states = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]]
    curr_states = []
    # find curr index of each number using .index(num) and find distance using respective goal state
    for i in range(1,len(state)):
        curr_num_index = state.index(i)
        curr_states.append(goal_states[curr_num_index])

    #print(curr_states)
    #print(goal_states)
    for i in range(len(curr_states)):
        distance += abs(curr_states[i][0] - goal_states[i][0]) + abs(curr_states[i][1] - goal_states[i][1])
    
    return distance


"""
Given a state of the puzzle, represented as a single list of integers with
a 0 in the empty space, print to the console all of the possible successor states
"""
def print_succ(state):
    succs = []
    empty_tile_index = state.index(0)
    #print(empty_tile_index)

    # check move left tile into empty grid
    if empty_tile_index == 0 or empty_tile_index == 3 or empty_tile_index == 6:
        pass
    else:
        succ1 = state.copy()
        succ1[empty_tile_index], succ1[empty_tile_index-1] = succ1[empty_tile_index-1], succ1[empty_tile_index]
        succs.append(succ1)
    
    # check move right tile into empty grid
    if empty_tile_index == 2 or empty_tile_index == 5 or empty_tile_index == 8:
        pass
    else:
        succ2 = state.copy()
        succ2[empty_tile_index], succ2[empty_tile_index+1] = succ2[empty_tile_index+1], succ2[empty_tile_index]
        succs.append(succ2)

    # check move top tile into empty grid
    if empty_tile_index == 0 or empty_tile_index == 1 or empty_tile_index == 2:
        pass
    else:
        succ3 = state.copy()
        succ3[empty_tile_index], succ3[empty_tile_index-3] = succ3[empty_tile_index-3], succ3[empty_tile_index]
        succs.append(succ3)

    # check move bottom tile into empty grid
    if empty_tile_index == 6 or empty_tile_index == 7 or empty_tile_index == 8:
        pass
    else:
        succ4 = state.copy()
        succ4[empty_tile_index], succ4[empty_tile_index+3] = succ4[empty_tile_index+3], succ4[empty_tile_index]
        succs.append(succ4)
    
    succs = sorted(succs)
    for succ in succs:
        manhattan_distance = calc_manhattan_distance(succ)
        print("{} h={}".format(succ,manhattan_distance))


def get_succs(state):
    succs = []
    empty_tile_index = state.index(0)
    #print(empty_tile_index)

    # check move left tile into empty grid
    if empty_tile_index == 0 or empty_tile_index == 3 or empty_tile_index == 6:
        pass
    else:
        succ1 = state.copy()
        succ1[empty_tile_index], succ1[empty_tile_index-1] = succ1[empty_tile_index-1], succ1[empty_tile_index]
        succs.append(succ1)
    
    # check move right tile into empty grid
    if empty_tile_index == 2 or empty_tile_index == 5 or empty_tile_index == 8:
        pass
    else:
        succ2 = state.copy()
        succ2[empty_tile_index], succ2[empty_tile_index+1] = succ2[empty_tile_index+1], succ2[empty_tile_index]
        succs.append(succ2)

    # check move top tile into empty grid
    if empty_tile_index == 0 or empty_tile_index == 1 or empty_tile_index == 2:
        pass
    else:
        succ3 = state.copy()
        succ3[empty_tile_index], succ3[empty_tile_index-3] = succ3[empty_tile_index-3], succ3[empty_tile_index]
        succs.append(succ3)

    # check move bottom tile into empty grid
    if empty_tile_index == 6 or empty_tile_index == 7 or empty_tile_index == 8:
        pass
    else:
        succ4 = state.copy()
        succ4[empty_tile_index], succ4[empty_tile_index+3] = succ4[empty_tile_index+3], succ4[empty_tile_index]
        succs.append(succ4)
    
    return sorted(succs)

"""
Given a state of the puzzle, perform the A* search algorithm and print
the path from the current state to the goal state
"""
def solve(state):
    final_result = []
    final_state = 0
    pq = [] # open
    closed = []
    goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    g_val = 0 # cost g
    h_val = calc_manhattan_distance(state) # cost h
    parent_index = -1 # keeps track of parent index
    visited = set()

    heapq.heappush(pq, (g_val + h_val, state, (g_val, h_val, parent_index)))
    

    while pq: # step 2
        #print(len(pq))
        n = heapq.heappop(pq)
        closed.append(n) # step 3
        """if n[2] in visited:
            continue
        else:
            visited.add(n[2])"""
        parent_index += 1
        #print(n)
        if n[1] == goal_state:
            # exit
            goal_state = n
            break # step 4
        else:
            #print("got here")
            succs = get_succs(n[1])

            for succ in succs:
                #print(succ)
                #print(closed)
                # if 1 + n[2][0] < 
                if (succ not in closed):
                    #print("added")
                    h_val = calc_manhattan_distance(succ)
                    heapq.heappush(pq, (1 + h_val + n[2][0], succ, (1 + n[2][0], h_val, parent_index)))
            
    final_result.append(goal_state)
    while goal_state[2][2] != -1:
        goal_state = closed[goal_state[2][2]]
        final_result.insert(0, goal_state)
    
    #print(visited)
    moves = 0
    for i in range(len(final_result)):
        print("{} h={} moves: {}".format(final_result[i][1],calc_manhattan_distance(final_result[i][1]), moves))
        moves += 1
          



