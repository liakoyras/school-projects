module find_winner(
	input logic [6:0][5:0][1:0] panel,
	input logic turn

	output logic winner,
	output logic exists
);

// when it is the turn of a player, we check if the other player made a winning move on the previous turn
logic [1:0] player;
always_comb begin
	if(turn==1'b0) begin
		player = 2'b10;
	end else begin
		player = 2'b01;
	end
end


always_comb begin
	exists = 0; // flag that is set if a winner is found
	
	// checking each row for a win
	for(int i=0; i<6; i=i+1) begin 
		for(int j = 0; j<4; j=j+1) begin
			if(exists == 0 && panel[j][i] == player && panel[j+1][i] == player && panel[j+2][i] == player && panel[j+3][i] == player) begin
				exists = 1; // confirming winner exists
				winner = ~turn; // winner is the previous player
			end
		end
	end
	
	// checking each column for a win
	for(int k=0; k<7; k=k+1) begin 
		for(int l = 0; l<3; l=l+1) begin
			if(exists == 0 && panel[k][l] == player && panel[k][l+1] == player && panel[k][l+2] == player && panel[k][l+3] == player) begin
				exists = 1; // confirming winner exists
				winner = ~turn; // winner is the previous player
			end
		end
	end
	
	// checking downwards (\) diagonals	for a win
	for(int m=0; m<4; m=m+1) begin
		for(int n = 0; n<3; n=n+1) begin
			if(exists == 0 && panel[m][n] == player && panel[m+1][n+1] == player && panel[m+2][n+2] == player && panel[m+3][n+3] == player) begin
				exists = 1; // confirming winner exists
				winner = ~turn; // winner is the previous player
			end
		end
	end
	
	// checking upwards (/) diagonals for a win
	for(int p=0; p<4; p=p+1) begin
		for(int r = 3; r<6; r=r+1) begin
			if(exists == 0 && panel[p][r] == player && panel[p+1][r-1] == player && panel[p+2][r-2] == player && panel[p+3][r-3] == player) begin
				exists = 1; // confirming winner exists
				winner = ~turn; // winner is the previous player
			end
		end
	end
	
end

endmodule