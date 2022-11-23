import numpy as np 

# Python Program to find the factors of a number

# define a function
def print_factors(x):
   """This function takes a
   number and prints the factors"""

   print("The factors of",x,"are:")
   for i in range(1, x + 1):
       if x % i == 0:
           print(i)

def get_factor(x):
   out = []
   for i in range(1, x + 1):
       if x % i == 0:
           out.append(i)
   return np.array(out)


if __name__ == '__main__':
# take input from the user
    num = int(input("Enter a number: "))

    print_factors(num)
