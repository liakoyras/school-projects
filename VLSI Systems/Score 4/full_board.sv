module full_board (
	input logic [6:0][5:0][1:0] panel,

	output logic full
);

logic [6:0] column_full;
always_comb begin
	column_full = 0;
	for(int i=0; i<7; i=i+1) begin // loops through the columns
		column_full[i] = |panel[i][0]; // creates a vector with 1 if full, 0 if empty
	end
	full = &column_full; // if all columns are full, the board is full
end


endmodule