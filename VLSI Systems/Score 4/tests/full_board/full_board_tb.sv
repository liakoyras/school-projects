module full_board_tb;

logic [6:0][5:0][1:0] panel;

logic full;


full_board fb (.panel(panel),
			.full(full)
			);

initial begin
	panel = 0;
	#5ns
	for(int i=0; i<7; i=i+1) begin // not full
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
	#5ns
	for(int i=0; i<7; i=i+1) begin // not full
		for(int j=0; j<6; j=j+1) begin
			panel[i][j] = 2'b01;
		end
	end
end


endmodule