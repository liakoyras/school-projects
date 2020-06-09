module free_row_tb;

logic [6:0][5:0][1:0] panel;
logic [6:0] play;

logic [2:0] free;
logic valid;


free_row row (.play(play),
			.panel(panel),
			.free(free),
			.valid(valid)
			);

initial begin
	for(int i=0; i<7; i=i+1) begin
		if(i==0) begin
			for(int j=0; j<6; j=j+1) begin
				panel[i][j] = 2'b01;
			end
		end else if(i==1) begin
			for(int j=0; j<6; j=j+1) begin
				if(j<3)
					panel[i][j] = 2'b00;
				else
					panel[i][j] = 2'b01;
			end
		end else begin
			for(int j=0; j<6; j=j+1) begin
				panel[i][j] = 2'b00;
			end
		end
	end
	play = 7'b0000001; //valid = 0, free=x
	#5ns
	play = 7'b0000010; // valid = 1, free=2
	#5ns
	play = 7'b0000001; //valid = 0, free=x
end


endmodule