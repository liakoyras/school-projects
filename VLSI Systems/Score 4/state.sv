module state (
	input  logic clk,
	input  logic rst,

	output logic [6:0][5:0][1:0] panel,
	output logic [6:0] play,
	output logic turn
);

// initial board state to test the vga output
// comment out if you want to test other functionalities
initial begin
	turn = 0; // red player next
	play = 7'b0000001; // 1st column selected
	for(int i=0; i<7; i=i+1) begin
		for(int j=0; j<6; j=j+1) begin
			if(i==3 && j==5) begin
				panel[i][j] = 2'b01; // red square in the middle column
			end else if(i==3 && j==4) begin
				panel[i][j] = 2'b10; // green square above it
			end else begin
				panel[i][j] = 2'b00;
			end
		end	
	end
end

endmodule