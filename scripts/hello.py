# def yielder():
#     loop = 0
#     while loop < 10:
#         yield "Hello", loop
#         loop += 1
#     yield "World", loop

# for s, i in yielder():
#     if s == "World":
#         print(i)
import sys
SIZE = 32

# set size to command line argument if provided
if len(sys.argv) > 1:
    SIZE = int(sys.argv[1])

print(SIZE)