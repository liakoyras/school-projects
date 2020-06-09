module find_winner_tb;

logic [6:0][5:0][1:0] panel;
logic turn;

logic winner;
logic exists;


find_winner find_win (.panel(panel),
			.turn(turn),
			.winner(winner),
			.exists(exists)
			);

initial begin
	turn = 1; // green player move, red has played already
	for(int i=0; i<7; i=i+1) begin
		for(int j=0; j<6; j=j+1) begin // diagonal
			if(i==1 && j==4) begin
				panel[i][j] = 2'b01;
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
	#5ns
	turn = 0; // red player move, green has played already
	for(int i=0; i<7; i=i+1) begin
		for(int j=0; j<6; j=j+1) begin // diagonal
			if(i==1 && j==4) begin
				panel[i][j] = 2'b10;
			end else if(i==2 && j==3) begin
				panel[i][j] = 2'b10; 
			end else if(i==3 && j==2) begin
				panel[i][j] = 2'b10; 
			end else if(i==4 && j==1) begin
				panel[i][j] = 2'b10; 
			end else begin
				panel[i][j] = 2'b00;
			end
		end	
	end
	#5ns
	turn = 1;
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
	turn = 0;
	for(int i=0; i<7; i=i+1) begin
		for(int j=0; j<6; j=j+1) begin // diagonal
			if(i==1 && j==4) begin
				panel[i][j] = 2'b10;
			end else begin
				panel[i][j] = 2'b00;
			end
		end	
	end
end


endmodule