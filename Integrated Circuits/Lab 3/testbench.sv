module testbench;

// Inputs
logic [8:0] X;
logic [8:0] O;

// Outputs
logic Error;
logic Full;
logic WinX;
logic WinO;
logic NoWin;


tictactoe ttt (
	.x (X),
	.o (O),
	.error (Error),
	.full (Full),
	.winX (WinX),
	.winO (WinO),
	.noWin (NoWin)
);
	
/* error er (
	.X(X),
	.O(O),
	.err(Error)
); */

/* logic [8:0] Temp;
full f (
	.X(X),
	.O(O),
	.isFull(Full),
	.temp(Temp)
);
 */
initial begin

	X = 9'b000000000; // Init
	O = 9'b000000000;
	
	#10ns;
	
	X = 9'b011100101; //Set starting values
	O = 9'b100011010;
	
	#10ns; // we are expecting output to be full, noWin
	
	O = 9'b110011010;
	
	#10ns; // we are expecting output to be full, error
	
	X = 9'b000000111; //Set starting values
	O = 9'b101000000;
	
	#10ns; // we are expecting winX
	
	
end
endmodule