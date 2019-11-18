module tictactoe (
input logic [8:0] x,
input logic [8:0] o,
output logic error,
output logic full,
output logic winX,
output logic winO,
output logic noWin
);

error e (.X[8:0](x[8:0]), .O[8:0](o[8:0]), .err(error));
full f (.X[8:0](x[8:0]), .O[8:0](o[8:0]), .isFull(full));
winX w0 (.X[8:0](x[8:0]), .win(winX));
winO w1 (.O[8:0](o[8:0]), .win(winO));

asign noWin = full && ~WinX && ~win0;

endmodule