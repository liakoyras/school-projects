module full_board(
	input logic [6:0][5:0][1:0] panel,

	output logic full
);

always_comb begin
	full = 1; // we are considering the board to be full until we find an empty spot
	for(int i = 0; i<7; i=i+1) begin // loops through the columns 
		if(panel[i][0] == 2'b00 && full) begin // checks if the last row is free (only if no columns with free spots were found so far)
			full = 0; // if a free spot is found, the board is not full anymore
		end
	end
end

endmodule