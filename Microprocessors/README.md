# Microprocessors
The ARM Assembly project I did for my 7th semester Microprocessors course.

## Description
I wrote an Assembly function that takes two numbers as input, along with a third number signifying one of the four basic arithmetic operations (addition, subtraction, multiplication and division).<br>
It then saves these numbers and the result of the operation on consecutive memory positions and then isolates each 4-bit nibble (representing a hexadecimal digit) and saves into the memory each of the 8 hex digits, and then encodes them in order to be displayed with 8 7-segment displays.

The file contains the function along with some example driver code in C.


The project was designed to run on an ARM Cortex-M0+ processor, tested using the FRDM-KL25Z pack on Keil µVision.
