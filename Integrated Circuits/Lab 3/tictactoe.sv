module tictactoe (
input logic [8:0] x,
input logic [8:0] o,
output logic error,
output logic full,
output logic winX,
output logic winO,
output logic noWin
);

error e (.X(x), .O(o), .err(error));
full f (.X(x), .O(o), .isFull(full));
winX w0 (.X(x), .win(winX));
winO w1 (.O(o), .win(winO));

assign noWin = ~winX && ~winO;

endmodule