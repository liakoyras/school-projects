# Integrated Circuits
The System Verilog projects I did for my 7th semester Integrated Circuits course and labs.

More specifically:

## Project 4 (Combinational logic)
We need to design:
- A circuit that will convert an one-hot encoded 8-bit signal to "thermometer" encoding
- A 10 to 1 multiplexer using 4 to 1 multiplexers

## Lab 3 (Combinational logic)
We need to design a circuit that represents the state of a tic-tac-toe game.<br>
It will have as inputs a one hot encoded representation of the Xs and Os and will calculate 5 output signals, indicating if an error has occurred (a box has both an X and an O), if there is no empty spot on the grid, if X won, if O won and if the game is still undergoing.

## Project 5 (Sequential logic-Testbench)
We need to design a simple 4-bit counter that has an extra control signal which signifies whether the counting will be continued with overflow (... 14, 15, 0, 1 ...) or the counter will stop at 11 (... 9, 10, 11, 11, 11 ...).

## Lab 4 (Verification)
We need to create a testbench in order to find out the bugs within an encrypted round robin arbiter module.<br>
The bugs can be controllably added by twitching a parameter.

## Project 8 (FSMD-Datapath)
We need to design a hardware unit that implements binary search in order to search a memory for a given value.<br>
We have at our disposal an addition/subtraction unit and a comparison unit, while the memory access is simplified.<br>
We need to provide the FSMD, it's description in System Verilog and the datapath that our FSMD drives.

## Lab 5 (Advanced design)
We need to design a round robin circuit for 8 possible requests, using the following components:
- An arbiter that decides which request will be granted and outputs the position of the granted request in weighted binary
- A priority register that holds the value of the request with the lowest priority in weighted binary
- One 4 bit to 7 segment display decoder (used to display the number of requests)
- One 3 bit to 7 segment display decoder (used to display the priority register's output)
- One 3 bit to 8 decoder, which converts the weighted binary of the granted request to one-hot (in order to display it on LEDs)

In addition, we need to match the outputs of the design to the proper pins on an Cyclone II EP2C20 FPGA (inside the .qsf file).
