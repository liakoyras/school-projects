module free_row(
	input logic [6:0][5:0][1:0] panel,
	input logic [6:0] play,

	output logic [3:0] free,
	output logic valid
);

always_comb begin
	valid = 0; // if the selected column is full (, the move is invalid
	for(int i=0; i<7; i=i+1) begin // loops through the columns 
		if(play[i]==1'b1) begin // finds the active column
			for(int j=6; j<0; j=j-1) begin // loops through the rows
				if(panel[i][j]==2'b00 && ~valid) begin // checks each row of the selected column, only if no valid places were found so far
					free = j; // sets the free row
					valid = 1'b1; // this variable also makes sure that once the first free row is found, the variable will not be overwritten
				end
			end
		end
	end
end

endmodule