module free_row (
	input logic [6:0][5:0][1:0] panel,
	input logic [6:0] play,

	output logic [2:0] free,
	output logic valid
);

// find the selected column
logic [2:0] selected_column;
active_column column (.play(play),
					.column(selected_column)
					);
					
logic [5:0] full_rows;
always_comb begin
	full_rows = 0;
	for(int i=0; i<6; i=i+1) begin // loops through the rows, starting from the bottom
		full_rows[i] = |panel[selected_column][i]; // creates a vector with 1 if full, 0 if empty
	end
	valid = ~(&full_rows); // if all columns are not full, it is a valid move
	
	if(full_rows==6'b000000) free = 5; // finds the position of the first 0
	else if(full_rows==6'b100000) free = 4;
	else if(full_rows==6'b110000) free = 3;
	else if(full_rows==6'b111000) free = 2;
	else if(full_rows==6'b111100) free = 1;
	else if(full_rows==6'b111110) free = 0;
	else free = 6;
end

endmodule