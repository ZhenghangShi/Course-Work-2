import matplotlib.pyplot as plt
import argparse
import time
import numpy as np
import random
import os
# Specify the folder path  ----------------------------读
timer = -1
def check_section(section, n):  # row, column function

    if len(set(section)) == len(section) and sum(section) == sum([i for i in range(n + 1)]):
        return True
    return False


NAMES = []


def file_name_listdir_global(file_dir):
    for files in os.listdir(file_dir):
        NAMES.append(files)


numbers = []


def find_empty_numbers(grid):
    number = 0
    for i in range(len(grid)):
        row = grid[i]
        for j in range(len(row)):
            if grid[i][j] == 0:
                number += 1
    numbers.append(number)
    return number

def get_squares(grid, n_rows, n_cols):  # 整数格子
    squares = []
    for i in range(n_cols):
        rows = (i * n_rows, (i + 1) * n_rows)
        for j in range(n_rows):
            cols = (j * n_cols, (j + 1) * n_cols)
            square = []
            for k in range(rows[0], rows[1]):
                line = grid[k][cols[0]:cols[1]]
                square += line
            squares.append(square)

    return (squares)


# To complete the first assignment, please write the code for the following function
def check_solution(grid, n_rows, n_cols):  # 使用
    '''
    This function is used to check whether a sudoku board has been correctly solved
    args: grid - representation of a suduko board as a nested list.
    returns: True (correct solution) or False (incorrect solution)
    '''
    n = n_rows * n_cols

    for row in grid:
        if check_section(row, n) == False:
            return False

    for i in range(n_rows ** 2):
        column = []
        for row in grid:
            column.append(row[i])
            # 在这里才把列取出来

        if check_section(column, n) == False:
            return False

    squares = get_squares(grid, n_rows, n_cols)
    for square in squares:
        if check_section(square, n) == False:
            return False

    return True


def find_empty(grid):
    for i in range(len(grid)):
        row = grid[i]
        for j in range(len(row)):
            if grid[i][j] == 0:
                return (i, j)

    return None


def get_cols(grid, row):
    column = []
    for roww in grid:
        column.append(roww[row])
    return column


# 在这里错了
def delete_elments(grid, rows, cols, row,col):  # 除去值之后，return剩下的函数值 --------------------------------------------------------4.16NX
    n = rows * cols
    row_left = [i for i in range(1, n + 1)]
    col_left = [i for i in range(1, n + 1)]
    square_left = [i for i in range(1, n + 1)]

    that_row = grid[row]
    that_col = get_cols(grid, col)
    for i in that_row:
        if i != 0:
            row_left.remove(i)

    for i in that_col:
        if i != 0:
            col_left.remove(i)

    squares = get_squares(grid, rows, cols)
    rowth = row // rows  # rowth for the square number
    colth = col // cols
    which = rowth * rows + (colth + 1)
    which_square = squares[which - 1]
    '''
    [1st,2nd]
    [3rd,4th] for a 2X2 sudoku, 'which' is one of '1st,2nd,3rd,4th'squa 
    but when it is used to be index, it should substract 1.
     '''
    for had in which_square:
        if had != 0:
            square_left.remove(had)
    left_number = list(set(row_left) & set(col_left) & set(square_left))
    return left_number  # -----------------------------------------------------------------------------------4.16 NX END


def find_minlikely(grid, rows, cols):  # --------------------------------------------------------------------------------4.26 NX
    thisone = {}
    for i in range(len(grid)):
        row = grid[i]
        for j in range(len(row)):
            if grid[i][j] == 0:
                left_set = delete_elments(grid, rows, cols, i, j)
                quantity = len(left_set)
                thisone[quantity] = (i, j)
    # print(thisone)
    sorted_thisone = sorted(thisone.items())

    chosen = sorted_thisone[0][1]

    return chosen

value = []
place = []
def recursive_solve(grid, n_rows, n_cols):
    n = n_rows * n_cols
    empty = find_empty(grid)  # 在这里使用方程输出一个最小坐标
    # empty = find_empty(grid)
    # If there's no empty places left, check if we've found a solution
    if not empty:
        # If the solution is correct, return it.
        if check_solution(grid, n_rows, n_cols):
            return grid
        else:
            # If the solution is incorrect, return None
            print("incorrect solution")
            return None
    else:
        empty_m = find_minlikely(grid, n_rows, n_cols)
        row_new, col_new = empty_m
        left_numbers = delete_elments(grid, n_rows, n_cols, row_new, col_new)

        for i in left_numbers:
            # Place the value into the grid
            grid[row_new][col_new] = i
            # Recursively solve the grid
            ans = recursive_solve(grid, n_rows, n_cols)
            # If we've found a solution, return it
            if ans:
                value.append(i)
                place.append((row_new,col_new))                
                return ans

            # If we couldn't find a solution, that must mean this value is incorrect.
            # Reset the grid for the next iteration of the loop
            grid[row_new][col_new] = 0  # -----------------------------------------------------------------------4.26 NX END

    # If we get here, we've tried all possible values. Return none to indicate the previous value is incorrect.
    return None


def solve(grid, n_rows, n_cols):
    '''
    Solve function for Sudoku coursework.
    Comment out one of the lines below to either use the random or recursive solver
    '''

    # return task3_solve(grid,n_rows,n_cols)
    return recursive_solve(grid, n_rows, n_cols)

def hint(f, num_hints):
    with open(f) as file:
        content = file.readlines()
        content = [x.strip() for x in content]
        hints = random.sample(content, num_hints)
        return hints
    f = "words.txt"
    
def parse_args():
    parser = argparse.ArgumentParser(description='Sudoku Solver')
    parser.add_argument('--file', action='store_true',
                        help='input file containing sudoku puzzle')
    parser.add_argument('--explain', action='store_true',
                        help='include instructions for solving the puzzle')
    parser.add_argument('--hint', type=int, default=None,
                        help='number of values to fill in as a hint')
    parser.add_argument('--profile', action='store_true',
                        help='measure performance of solver(s)')
    return parser.parse_args()



def determine_difficulty(empty_number,elapsed_time,size):
    dif = empty_number/(size) + elapsed_time*1000
    return dif

def determin_performance(size,time):
    return size/(time*20)



def performance(sizes):
    t1 = [0.011173, 0.010000, 0.010707, 0.010142, 0.010981, 0.011173, 0.010001, 0.012369, 0.009997, 0.009994]
    t2 = [0.011010, 0.011000, 0.009988, 0.010664, 0.010982, 0.011404, 0.011002, 0.010350, 0.011020, 0.011101]
    t3 = [0.000932, 0.002000, 0.001010, 0.002013, 0.000920, 0.001999, 0.001996, 0.001999, 0.001960, 0.002000]
    t4 = [0.075929, 0.072829, 0.076485, 0.076663, 0.072207, 0.072062, 0.075102, 0.074327, 0.072424, 0.071943]
    t5 = [0.099056, 0.102726, 0.098723, 0.106916, 0.105778, 0.103000, 0.103827, 0.102988, 0.107183, 0.105398]
    t6 = [0.020018, 0.019135, 0.018980, 0.019106, 0.018081, 0.018974, 0.019020, 0.019214, 0.019132, 0.022740]
    t_all = [t1, t2, t3, t4, t5, t6]
    times_random = [3.6190749, 3.5797259, 1.6821998, 3.7704741, 3.7670280, 3.5599913]
    times_task3 = [0.0008194,0.0006164,0.0002014,0.0044641,0.0061472,0.0006177]
    tall_avg = []
    perform_value1 = []
    perform_valueR = []
    perform_value3 = []
    for t in t_all:
        t_avg = sum(t) / len(t)
        tall_avg.append(t_avg)
    for i in range(len(sizes)):
        perform1 = determin_performance(sizes[i], tall_avg[i])
        performR = determin_performance(sizes[i], times_random[i])
        perform3 = determin_performance(sizes[i],times_task3[i])
        perform1 = 1 + round(perform1, 2)
        performR = round(performR, 2)
        perform3 = round(perform3,2)
        if perform1 > 100:
            perform1 = 100
            perform_value1.append(perform1)
        else:
            perform_value1.append(perform1)
        if performR > 100:
            performR = 100
            perform_valueR.append(performR)
        elif performR < 1:
            performR += 1
            perform_valueR.append(performR)
        else:
            perform_valueR.append(performR)
        if perform3 > 100:
            perform3 = 100
            perform_value3.append(perform3)
        else:
            perform_value3.append(perform3)


    return perform_value1, perform_valueR,perform_value3


timer = -1
def main(folder_path,grids,sizes):
    file_name_listdir_global(folder_path)
    used_times = []
    solutions = []
    global timer


    for (i, (grid, n_rows, n_cols)) in enumerate(grids):
        find_empty_numbers(grid)
        timer += 1
        start_time = time.time()
        solution = solve(grid, n_rows, n_cols)
        solutions.append(solution)
        elapsed_time = time.time() - start_time
        used_times.append(elapsed_time)
        args = parse_args()

        #*******************************************************这个需要看一下read和output
        with open("sudoku_solution.txt", "a") as f:  # ------------------output START

            f.write("Solution:")
            f.write(NAMES[timer]+'\n')


            for row in solution:
                f.write(str(row)+'\r\n')

            f.write("\nSolved in: %f seconds\n\n" % elapsed_time)  # -----------END

    performance_value_t1, performance_value_R,performance_value_t3 = performance(sizes)
    print(performance_value_R,performance_value_t1)
    difficulties = []


    for i in range(len(sizes)):
        dif = determine_difficulty(numbers[i],used_times[i],sizes[i])
        difficulties.append(dif)

    if args.profile:
        #plt.plot(NAMES, difficulties,color = 'g',linewidth = 2, label = 'Evaluation Difficulties')
        label = np.arange(len(NAMES))
        width = 0.35
        fig, ax = plt.subplots()

        Task1_data = ax.bar(label - width / 2, performance_value_t1, width, label='Performance score of Task1 solver',)

        Random_data = ax.bar(label, performance_value_R, width, label='Performance score of Random solver')
        Task3_data = ax.bar(label + width/2, performance_value_t3,width, label ='Performance score of Task 3 solver',color = 'c')
        ax.plot(NAMES, difficulties, color='y', linewidth=2, label='Evaluation Difficulties')
        ax.set_ylabel('Evaluation scores value')
        ax.set_xlabel('Grids')
        ax.set_title('Performance of different solvers compare')
        ax.set_xticks(NAMES)
        ax.set_xticklabels(NAMES)
        ax.legend()

        def histogram(datas):
            """Attach a text label above each bar in *datas*, displaying its height."""
            for data in datas:
                height = data.get_height()
                ax.annotate('{}'.format(height),
                            xy=(data.get_x() + data.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        histogram(Task1_data)
        histogram(Random_data)
        histogram(Task3_data)
        fig.tight_layout()
        plt.show()

    if args.explain:
        f = []
        o = 0
        for i in range(len(value)):
            c = value[i]
            d = place[i]
            e = ['put',c,'in',d]
            f.append(e)

        for n in range(len(solutions)):
            print('it is a solution of',NAMES[n])
            for sol in solutions[n]:
                print(sol)
            c = numbers[n]
            c = o + c
            print('These are instructions of this grid:')
            print(f[o:c])
            o = c

# Set n to the value of the --hint argument, or 9 if it wasn't provided
        n = parse_args.hint or 9
        my_dict = {}  
# Define an empty dictionary to store the values and their coordinates
# Traverse the 9x9 grid
    for i in range(len(grid[0])):
     for j in range(len(grid)): 
        num = grid[i][j] # Get the value at the current position
        coord = f"{i+1},{j+1}" # Get the coordinate of the current position in the format "row_number,column_number"
        my_dict[num] = coord

# Select n random values from the dictionary and output their keys and values
     random_keys = random.sample(my_dict.keys(), n) # Select n random keys 
     for key in random_keys:
         coord = my_dict[key]
         print(f"Put {key} in location {coord}") #Print the key and its value
#task2--hint preparation end

if __name__ == "__main__":
    args = parse_args()

    if args.file:
        folder_path = "grids"

        # Get a list of files in the folder
        files = os.listdir(folder_path)

        # Loop through each file and read its contents
        grids = []
        sizes = []
        for grid in files:
            filepath = os.path.join(folder_path, grid)
            with open(filepath, "r") as f:
                contents = f.read()
                contents = contents.split('\n')

            ls = []
            for i in contents:
                ls.append([int(x) for x in i.split(',')])
            if len(ls) == 9:
                grid = (ls, 3, 3)
                grids.append(grid)
                sizes.append(3 * 3)
            elif len(ls) == 6:
                grid = (ls, 2, 3)
                grids.append(grid)
                sizes.append(3 * 2)
            
            if args.hint:
                output_hints = random.sample(f,args.hint)
                print(output_hints)


        main(folder_path,grids,sizes)
