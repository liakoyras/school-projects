module state (
	input  logic clk,
	input  logic rst,

	output logic [6:0][5:0][1:0] panel,
	output logic [6:0] play,
	output logic turn
);

assign turn = 1; // green

assign play = 7'b0000001; // 5th collumn
/* assign play[4] = 1;  */


initial begin
	for(int i=0; i<7; i=i+1) begin
		for(int j=0; j<6; j=j+1) begin
			if(i==3 && j==5) begin
				panel[i][j] = 2'b01;
			end else begin
				panel[i][j] = 2'b00;
			end
		end	
	end
end

endmodule