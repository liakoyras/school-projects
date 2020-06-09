module state (
	input  logic clk,
	input  logic rst,
	
	input logic [6:0][5:0][1:0] panel_in,
	input logic [6:0] play_in,
	input logic turn_in,

	output logic [6:0][5:0][1:0] panel,
	output logic [6:0] play,
	output logic turn
);

// initializing to an empty board
initial begin
	turn = 0; // red player move
	play = 7'b0000001; // 1st column selected
	panel = 0; // empty board
end

// registers for the game state variables
always_ff @(posedge clk, posedge rst) begin
	if(rst) begin
		panel <= 0;
		play <= 7'b0000001;
		turn <= 0;
	end else begin
		panel <= panel_in;
		play <= play_in;
		turn <= turn_in;
	end
end


/* initial board state to test the vga output */
/* comment out if you want to test other functionalities */

// simple test
/* initial begin
	turn = 0; // red player move
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
end */

// test check winner
/* initial begin
	turn = 1; // green player move
	play = 7'b0000001; // 1st column selected
	for(int i=0; i<7; i=i+1) begin
		for(int j=0; j<6; j=j+1) begin
			if(i==1 && j==4) begin
				panel[i][j] = 2'b01; // 4 red squares on the bottom
			end else if(i==2 && j==3) begin
				panel[i][j] = 2'b01; 
			end else if(i==3 && j==2) begin
				panel[i][j] = 2'b01; 
			end else if(i==4 && j==1) begin
				panel[i][j] = 2'b01; 
			end else begin
				panel[i][j] = 2'b00;
			end
		end	
	end
end */


endmodule