# Python program to create aes_file for convenience. This is highly INSECURE
# Pass maximum number of parties and party_num as arguments (party_num index starts from 0)
# Current file allows at most 5PC code
# 
# For instance running the following command
#       python3 aes_setup.py 5 3 
# will create a file key3 which contains the aes keys for party_num = 3 for at most 5-PC
# key3 will contain 5 rows, where ith row is key shared with partyu_num i (thus 4th entry is local randomness key)



# importing os module  
import os 
import sys

# total arguments
num_args = len(sys.argv)
assert num_args == 2 or num_args == 3 
assert int(sys.argv[1]) < 6
num_parties = int(sys.argv[1])


# path to keys file
path = '../files/keys'

# Create the directory 
try: 
    os.mkdir(path) 
except OSError as error: 
    print(error)



# create all associated files
def main(): 
    if (num_args == 2):
        for i in range(num_parties):
            create_keyfile(i)
    elif (num_args == 3):
        assert int(sys.argv[2]) < int(sys.argv[1])
        create_keyfile(int(sys.argv[2]))
    else:
        print("Invalid number of arguments passed to: ", sys.argv[0]);




# helper functions
def create_keyfile(party_num):
    with open(path + "/key" + str(party_num), 'wt', encoding='utf-8') as file:
        for i in range(num_parties):
            file.write(global_keys[get_index(i, party_num)])
            file.write('\n')

def get_index(i, j):
    if (min(i, j) == max(i, j)):
        return i * (i+1) // 2
    else: 
        return (max(i, j) * (max(i, j) + 1) // 2) + 1 + min(i, j)

global_keys = [
    '8E48D38503CF26B73DC1A9C63BA3F336',
    'C038BDA962B7424B5098A503804E49FE',
    'D3K4M5P7Q8R2J3K4N6P7Q9S2J4M5N6QS',
    '0R0R6EZTHZRQ9YP8BOJ37BOFKABIGCK2',
    '4WWXO2VPHVGUU6VF9DKI3F0KY4BODEDQ',
    'KI11IHPOORP5B1NKUOMBAQ1I4H74HI9Y',
    '5IR9DVNEQMG2CS62DT7T4P4RLZKT7VVA',
    '9DAC8GWG6FEAJ15SF0DJEOKIOCPYE2EU',
    '73Q6QG980DZLUN8W9BGWFP7S4QIHIILW',
    'D920ZCSTNX0JUPMDMR8PEBE83AE7QWIV',
    '9LSSPR1BDHRJ2QH9TKB23G9DHCC8HCQ4',
    '2XV0NGX41ND8WARL220XTCDAHAT0H85V',
    '6ZFZMJKV69OO9MKQE17SRRE6HAN91882',
    'RIM2A6M4ZZEHNEWA17QI7FSS4M5EJ61I',
    'XFO74N7YH67GD2TWVXR5IZI6O1CNBLCB'
    ]

# The global_keys file mapping follows the following map
#   0,0     --> 0
#   1,1     --> 1
#   1,0/0,1 --> 2
#   2,2     --> 3
#   2,0/0,2 --> 4
#   2,1/1,2 --> 5
#   3,3     --> 6
#   3,0/0,3 --> 7
#   3,1/1,3 --> 8
#   3,2/2,3 --> 9
#   4/4     --> 10

if __name__ == '__main__':
    main()
